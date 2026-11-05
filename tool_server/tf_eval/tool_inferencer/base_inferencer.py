
import torch
from torch.utils.data import DataLoader,Dataset
from accelerate import Accelerator
import requests
import copy
import json
import os

from ..models.abstract_model import tp_model
from .dynamic_batch_manager import DynamicBatchManager, normalize_action_response, parse_single_action
from ..utils.utils import *
from ..utils.log_utils import get_logger
from ...tool_workers.tool_manager.base_manager import ToolManager
from ...tool_workers.cache_manager.base_cache_manager import CacheManager
from ...tool_workers.skills.registry import SKILL_REGISTRY
MAX_TOOL_TEXT_CHARS = 1200

import torch.distributed as dist

logger = get_logger(__name__)

class BaseToolInferencer(object):
    def __init__(
        self,
        tp_model: tp_model = None,
        # dataset: Dataset = None,
        batch_size: int = 1,
        model_mode: str = "general",
        max_rounds: int = 3,
        stop_token: str = "<stop>",
        controller_addr: str = "http://0.0.0.0:20001",
        cache_dir: str | None = None,
        save_dir: str | None = None,
        disable_skills: bool = False,
    ):
        self.accelerator = Accelerator()
        self.tp_model = tp_model
        self.model_mode = model_mode 
        self.generate_conversation_fn = self.tp_model.generate_conversation_fn
        self.append_conversation_fn = self.tp_model.append_conversation_fn

        if dist.is_initialized() and self.accelerator.device.type == "cuda" and not 'vllm_models' in str(type(self.tp_model)):
            self.tp_model = self.tp_model.to(self.accelerator.device)
            self.tp_model = self.tp_model.to(torch.bfloat16)

        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.stop_token = stop_token
        self.controller_addr = controller_addr
        self.manager = DynamicBatchManager(
            batch_size=self.batch_size, 
            max_rounds=self.max_rounds, 
            stop_token=self.stop_token,
            generate_conversation_fn = self.tp_model.generate_conversation_fn,
        )
        self.tool_manager = ToolManager()
        self.save_dir = save_dir
        self.disable_skills = disable_skills
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.cache_manager = CacheManager(cache_dir,save_dir)
        self.available_models = self.tool_manager.available_tools
        if self.disable_skills:
            self.available_models = [
                tool for tool in self.available_models
                if tool != "SkillExecutor" and tool not in SKILL_REGISTRY
            ]
        logger.info(f"{len(self.available_models)} available models: {self.available_models}")

    def _resolve_image_ref(self, images, ref=None, item_img=None, default_index=0):
        """Resolve dataset image aliases such as img_1/img_2/tif_1/tif_2."""
        first_aliases = {
            "img_1", "image_1", "tif_1", "primary_image", "$primary_image",
            "optical_image", "$optical_image", "time1_image", "$time1_image",
            "pre_image", "$pre_image",
        }
        second_aliases = {
            "img_2", "image_2", "tif_2", "sar_image", "$sar_image",
            "time2_image", "$time2_image", "post_image", "$post_image",
        }
        output_aliases = {"out.tif", "output_image", "$output_image"}

        if ref is not None and not isinstance(ref, str):
            ref = str(ref)

        if ref in output_aliases and item_img:
            return item_img, None
        if ref in first_aliases:
            index = 0
        elif ref in second_aliases:
            index = 1
        elif ref:
            return ref, None
        elif item_img:
            return item_img, None
        else:
            index = default_index

        if not images:
            return None, "Valid image is required for tool"
        if not isinstance(images, list) or not all(isinstance(img, str) for img in images):
            return None, f"Images must be a list of strings, got {type(images)}"
        if index >= len(images):
            return None, f"Image reference {ref or f'index {index}'} is unavailable for this sample"
        return images[index], None

    def _append_image_error(self, item, message):
        item.tool_response.append(dict(text=message, error_code=1))

    def _replace_last_assistant_message(self, item, content):
        if item.conversation and item.conversation[-1].get("role") == "assistant":
            item.conversation[-1]["content"] = content

    def _extract_thought(self, text):
        if not isinstance(text, str):
            return ""
        try:
            payload = json.loads(text.strip())
        except Exception:
            return ""
        thought = payload.get("thought") if isinstance(payload, dict) else ""
        return thought if isinstance(thought, str) else ""

    def _build_parse_feedback(self, action_parse_error):
        if action_parse_error == "Multiple_tool_calls_in_single_step":
            return "Your previous response had multiple actions. Return exactly one JSON object with one action."
        if action_parse_error == "No action found.":
            return "Your previous response had no valid action. Return exactly one JSON object with one action."
        return "Your previous response was invalid. Return exactly one valid JSON object with exactly one action."

    def _compact_tool_response(self, api_name, text, error_code=0):
        if text is None:
            text = ""
        text = str(text).strip()
        if error_code and len(text) > 300:
            return text[:300] + " ..."
        if api_name in {"ObjectDetection", "SmallObjectDetection", "DenoDetV2", "TextToBbox"}:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) > 8:
                return "\n".join(lines[:8]) + f"\n... ({len(lines) - 8} more detections omitted)"
        if len(text) > MAX_TOOL_TEXT_CHARS:
            return text[:MAX_TOOL_TEXT_CHARS] + " ..."
        return text

    def _validate_geotiff(self, geotiff):
        if not isinstance(geotiff, str) or not geotiff:
            return "GeoTIFF path is required."
        if not geotiff.lower().endswith((".tif", ".tiff")):
            return f"GeoTIFF tool requires a .tif/.tiff file, got {geotiff}."
        if not os.path.exists(geotiff):
            return f"GeoTIFF not found: {geotiff}."
        try:
            import rasterio
            with rasterio.open(geotiff) as src:
                if src.crs is None:
                    return f"GeoTIFF has no CRS: {geotiff}."
        except Exception as error:
            return f"Cannot load GeoTIFF {geotiff}: {error}"
        return None

    def batch_tool_response_to_next_round_input(self):
        current_batch = self.manager.get_current_batch()
        
        for idx,item in enumerate(current_batch):
            if item.model_response is None or item.status != "processing":
                continue
            # breakpoint() 
            tool_cfg = item.tool_cfg[item.current_round-1]
            tool_response = item.tool_response[item.current_round-1]
            action_parse_error = item.action_parse_error[item.current_round-1]
            action_parse_warning = item.action_parse_warning[item.current_round-1]

            assert len(item.tool_cfg) == item.current_round 
            assert len(item.tool_response) == item.current_round 
            assert len(item.action_parse_error) == item.current_round 
            assert len(item.action_parse_warning) == item.current_round

            if tool_response is not None:
                try:
                    if "gpkg" in tool_response:
                        item.gpkg = tool_response.pop("gpkg")
                    if "image" in tool_response:
                        item.img = tool_response.pop("image")
                   
                    if "text" in tool_response:
                        api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))
                        tool_response_text = self._compact_tool_response(
                            api_name,
                            tool_response["text"],
                            tool_response.get("error_code", 0),
                        )
                        ###Without feed back
                        # if tool_response["error_code"] == 0:
                        #     tool_response_text = tool_response["text"]
                        # else:
                        #     tool_response_text = "Could not be completed"
                    else:
                        tool_response_text = None
                    # breakpoint() 
                    warning_text = ""
                    if action_parse_warning:
                        warning_text = "\nNOTE: Only the first action from the previous response was executed."
                    new_response = f"OBSERVATION:\n{api_name}: {tool_response_text}{warning_text}\n"
                    new_round_prompt = f"{new_response}Next: call exactly one action or Terminate."
                except Exception as e:
                    logger.error(
                        f"[batch_tool_response_to_next_round_input] "
                        f"Failed to build new_round_prompt for tool"
                        f"Reason: {e}. Tool response: {str(tool_response)}"
                    )
                    new_round_prompt = "Tool response could not be converted into an observation. Next: call exactly one action or Terminate."
            else:
                if action_parse_error:
                    new_round_prompt = self._build_parse_feedback(action_parse_error)
                else:
                    new_round_prompt = "No tool was executed. Next: call exactly one action or Terminate."


            new_round_input = dict(text=new_round_prompt)
            item.new_round_input.append(new_round_input)
            item.conversation = self.append_conversation_fn(
                conversation=item.conversation, text=new_round_prompt, role="user"
            )
            # breakpoint()

    
    def batch_get_tool_response(self):
        current_batch = self.manager.get_current_batch()
        for item in current_batch:
            if item.model_response is None or item.status != "processing":
                continue
            
            tool_cfg = item.tool_cfg[item.current_round-1]
            assert len(item.tool_cfg) == item.current_round
            images = item.meta_data.get("images", None)
            if tool_cfg is not None and len(tool_cfg) > 0:
                assert item.status == "processing"
                try:
                    assert len(tool_cfg) == 1, "Only one tool is supported for now, but got: {}".format(tool_cfg)
                    api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))

                    if self.disable_skills and (api_name == "SkillExecutor" or api_name in SKILL_REGISTRY):
                        logger.info(f"API_name {api_name} is disabled in no-skill evaluation.")
                        item.tool_response.append(dict(
                            text=f"{api_name} is disabled in no-skill e2e evaluation. Use raw tools directly.",
                            error_code=1,
                        ))
                        continue

                    if api_name not in self.available_models:
                        if api_name == "Terminate":
                            logger.info(f"API_name is {api_name}. Finish!")
                            continue
                        else:
                            logger.error(f"API_name {api_name} not in available models, {self.available_models}")
                            item.tool_response.append(dict(text=f"There is no tool named {api_name}.",error_code=1))
                            continue

                    # if input contains image
                    IMAGE_REQUIRED_TOOLS = [
                        "DrawBox",
                        "AddText",
                        "TextToBbox",
                        "CountGivenObject",
                        "ImageDescription",
                        "RegionAttributeDescription",
                        "ChangeDetection",
                        "SegmentObjectPixels",
                        "ObjectDetection",
                        "SmallObjectDetection",
                        "DenoDetV2",
                        "GetBboxFromGeotiff",
                        "DisplayOnGeotiff"
                    ]
                    GPKG_REQUIRED_TOOLS = [
                        "AddPoisLayer",
                        "ComputeDistance",
                        "DisplayOnGeotiff"
                    ]
                    USE_CACHE_TOOLS = [
                        "GetAreaBoundary",
                        "AddPoisLayer",
                        "ComputeDistance",
                    ]
                    ARTIFACT_OUTPUT_TOOLS = {
                        "GetAreaBoundary",
                        "DisplayOnGeotiff",
                        "DrawBox",
                        "AddText",
                        "Plot",
                    }
                    image = pre_image = post_image = geotiff = gpkg = None
                    raw_api_params = tool_cfg[0].get("api_params", tool_cfg[0].get("API_params", {})).copy()

                    if api_name == "ChangeDetection":
                        pre_image, error = self._resolve_image_ref(
                            images,
                            raw_api_params.get("pre_image"),
                            default_index=0,
                        )
                        if error:
                            self._append_image_error(item, error)
                            continue
                        post_image, error = self._resolve_image_ref(
                            images,
                            raw_api_params.get("post_image"),
                            default_index=1,
                        )
                        if error:
                            self._append_image_error(item, error)
                            continue
                    elif api_name in {"DisplayOnGeotiff", "GetBboxFromGeotiff"}:
                        geotiff, error = self._resolve_image_ref(
                            images,
                            raw_api_params.get("geotiff") or raw_api_params.get("image"),
                            item_img=item.img,
                            default_index=0,
                        )
                        if error:
                            self._append_image_error(item, error)
                            continue
                        error = self._validate_geotiff(geotiff)
                        if error:
                            item.tool_response.append(dict(text=error, error_code=3))
                            continue
                    elif api_name in IMAGE_REQUIRED_TOOLS:
                        image, error = self._resolve_image_ref(
                            images,
                            raw_api_params.get("image"),
                            item_img=item.img,
                            default_index=0,
                        )
                        if error:
                            self._append_image_error(item, error)
                            continue
     
                    if api_name in GPKG_REQUIRED_TOOLS:
                        if not item.gpkg:
                            item.tool_response.append(dict(
                                text=f"GeoPackage is required before calling {api_name}.",
                                error_code=3,
                            ))
                            continue
                        gpkg = item.gpkg

                    api_params = raw_api_params.copy()
                    for key in ("image", "pre_image", "post_image", "gpkg", "geotiff"):
                        api_params.pop(key, None)

                    if api_name == "ChangeDetection":
                        api_params["pre_image"] = pre_image
                        api_params["post_image"] = post_image
                    if api_name == "DisplayOnGeotiff" or api_name == "GetBboxFromGeotiff":
                        api_params["geotiff"] = geotiff
                    if api_name in IMAGE_REQUIRED_TOOLS and api_name not in {"DisplayOnGeotiff", "GetBboxFromGeotiff"}:
                        api_params["image"] = image
                    if api_name in GPKG_REQUIRED_TOOLS:
                        api_params["gpkg"] = gpkg
                    if self.save_dir and api_name in ARTIFACT_OUTPUT_TOOLS:
                        api_params.setdefault("save_path", self.save_dir)
                        api_params.setdefault("output_dir", self.save_dir)

                    use_cache_successful = False
                    if api_name in USE_CACHE_TOOLS:
                        tool_response = self.cache_manager.get_response(item,api_name,api_params)
                        if tool_response['error_code'] == 0:
                            use_cache_successful = True
                            logger.info("Use cache successful")
                    if api_name not in USE_CACHE_TOOLS or not use_cache_successful:
                        tool_response = self.tool_manager.call_tool(api_name, api_params)
                    res_text = tool_response['text']    
                    if isinstance(res_text, str) and len(res_text) > MAX_TOOL_TEXT_CHARS:
                        tool_response['text'] =(
                            res_text[:MAX_TOOL_TEXT_CHARS] + " ... "
                        )  
                    tool_response_clone = copy.deepcopy(tool_response)

                    if  tool_response['error_code'] == 0:
                        logger.info(f"The {api_name} calls successfully!")
                    else:
                        logger.info(f"The {api_name} calls failed!")
                    item.tool_response.append(tool_response_clone)
                    continue
                except Exception as e:
                    # import traceback
                    # err_trace = traceback.format_exc()
                    logger.error(f"[Tool Error] {api_name} failed. Reason: {e}\n") 
                    item.tool_response.append(dict(text=f"Tool {api_name} failed : {e}", error_code=1))
                    continue
            else:
                item.tool_response.append(None)
                continue
            
    def extract_actions(self, text: str):
        action, action_parse_error = parse_single_action(text)
        if action is None:
            return None, action_parse_error
        return [action], None
       
    def batch_parse_tool_config(self):
        current_batch = self.manager.get_current_batch()
        for item in current_batch:
            model_response = item.model_response[item.current_round-1]
            assert len(item.model_response) == item.current_round
            # breakpoint()
            if model_response is None or item.status != "processing":
                continue
            tool_cfg = None
            action_parse_warning = None
            try:
                # breakpoint()
                if self.model_mode == "general":

                    actions, action_parse_error = self.extract_actions(model_response)
                    # breakpoint() 
                    if actions is not None:
                        action = actions[0]
                        assert 'name' in action and 'arguments' in action, "missing 'name' or 'arguments' in the parsed action."
                        if action_parse_error:
                            action_parse_warning = action_parse_error
                        action_parse_error = None
                        tool_cfg = [{'API_name': action['name'],
                                    'API_params': action['arguments']}]
                        self._replace_last_assistant_message(
                            item,
                            normalize_action_response(action, self._extract_thought(model_response)),
                        )
                    else:
                        tool_cfg = None
                        self._replace_last_assistant_message(item, "<invalid assistant response omitted>")

            except Exception as e:
                msg = f"Invalid action format."
                logger.info(f"{msg}: {e}")
                action_parse_error = msg
                tool_cfg = None
                self._replace_last_assistant_message(item, "<invalid assistant response omitted>")

            item.action_parse_error.append(action_parse_error)
            item.action_parse_warning.append(action_parse_warning)
            item.tool_cfg.append(tool_cfg)

    ## Batch Inference
    def batch_inference(self,dataset):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=2, 
            collate_fn=lambda x: x[0]
        )

        if dist.is_initialized() and not 'vllm_models' in str(type(self.tp_model)):
            self.dataloader = self.accelerator.prepare(self.dataloader)
        self.dataloader_iter = iter(self.dataloader)
        self.tp_model.eval()

        progress_bar = tqdm_rank0(len(self.dataloader), desc="Model Responding")

        if len(self.dataloader) == 0 and not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
            return
        self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)


        current_batch = self.manager.get_current_batch()
        self.tp_model.generate(current_batch)

        self.manager.update_item_status()
        while len(current_batch) > 0:
            try:
                # Inspect and yield output
                results = self.manager.pop_qualified_items()
                for res in results:
                    idx = res["meta_data"]["idx"]
                    self.dataset.store_results(dict(idx=idx,results=res))

                # Parse tool config and generate tool response
                self.batch_parse_tool_config()

                self.batch_get_tool_response()
                self.batch_tool_response_to_next_round_input()
                
                # Refill the current batch
                self.manager.append_item_to_full(self.dataloader_iter,progress_bar=progress_bar)
                
                current_batch = self.manager.get_current_batch()
                self.tp_model.generate(current_batch)
                self.manager.update_item_status()

            except StopIteration:
                break
        # breakpoint()
        assert len(self.manager.get_current_batch()) == 0
        if not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
    
    def convert_conv_to_msges(self, conv):
        messages = []
        for item in conv:
            if item["from"] == "human":
                clean_text = item["value"].replace('<AGENT_PROMPT>\n\nQuestion:', '').strip()
                self.append_conversation_fn( messages, text = clean_text, role = "user")
            if item["from"] == "gpt":
                self.append_conversation_fn(messages, text = item["value"] ,role = "assistant") 
        return messages
    
    def step_inference(self,dataset):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=2, 
            collate_fn=lambda x: x,
            shuffle=False,
        )

        if dist.is_initialized() and not 'vllm_models' in str(type(self.tp_model)):
            self.dataloader = self.accelerator.prepare(self.dataloader)
        self.tp_model.eval()

        if len(self.dataloader) == 0 and not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
            return
        
        if dist.is_initialized() and not 'vllm_models' in str(type(self.tp_model)):
            progress_bar = tqdm_rank0(len(self.dataloader), desc="Model Responding (step-mode)")
        else:
            progress_bar = tqdm_rank0(len(dataset), desc="Model Responding (step-mode)")

        for batch in self.dataloader:
            if progress_bar is not None:
                progress_bar.update(1 if dist.is_initialized() and not 'vllm_models' in str(type(self.tp_model)) else len(batch))

            inputs = []
            meta_data_batch = []
            for item in batch:
                item_meta_data = copy.deepcopy(item)
                item_meta_data["input"] = self.convert_conv_to_msges(item["text"])
                inputs.append(item_meta_data["input"])
                meta_data_batch.append(item_meta_data)

            model_outputs = self.tp_model.generate(inputs)
            for item_meta_data, model_output in zip(meta_data_batch, model_outputs):
                result = {}
                result["meta_data"] = item_meta_data
                result["model_output"] = model_output
                self.dataset.store_results(dict(idx=item_meta_data["idx"],results=result))
        
        if not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
    
    

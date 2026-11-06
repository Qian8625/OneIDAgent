import os
import requests

from ..offline_workers import offline_tool_workers, get_tool_generate_fn
from tool_server.utils.utils import load_json_file
from tool_server.utils.server_utils import build_logger
from contextlib import contextmanager
import signal
import time

CORE_EXPECTED_ONLINE_TOOLS = [
    'Calculator',
    'DrawBox',
    'AddText',
    'GoogleSearch',
    'Plot',
    'Solver',
    'TextToBbox',
    'BaseModel',
    'ImageDescription',
    'RegionAttributeDescription',
    'CountGivenObject',
    'SegmentObjectPixels',
    'ChangeDetection',
    'ObjectDetection',
    'GetAreaBoundary',
    'AddPoisLayer',
    'ComputeDistance',
    'SmallObjectDetection',
    'DenoDetV2',
    'CrossModalInterpret',
]

EXPECTED_ONLINE_TOOLS = CORE_EXPECTED_ONLINE_TOOLS

logger = build_logger("tool_manager")
class TimeoutException(Exception): pass
def _timeout_handler(signum, frame):
    raise TimeoutException("chat() timed out.")
signal.signal(signal.SIGALRM, _timeout_handler)

class ToolManager(object):
    def __init__(self, controller_url_location=None):
        self.controller_url_location = controller_url_location
        self.init_offline_tools()
        self.init_online_tools(self.controller_url_location)
        self.init_online_tool_addr_dict()
        logger.info(f"ToolManager is initialized.")
        self.internal_available_tools = self.available_offline_tools + self.available_online_tools

        self.agent_available_tools = self.available_offline_tools + self.available_online_tools

        self.available_tools = self.agent_available_tools
        self.headers = {"User-Agent": "Client"}
        
    def init_online_tools(self, controller_url_location=None):
        self.available_online_tools = []
        self.online_tools_init_error = None

        if controller_url_location is None:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = (
                f"{current_file_path}/../online_workers/controller_addr/controller_addr.json"
            )
            logger.info("controller_addr is None, using default from controller_addr_location")
        else:
            self.controller_addr_location = controller_url_location
            logger.info(f"controller_addr exists, controller_url_location is {controller_url_location}")

        if os.path.exists(self.controller_addr_location):
            try:
                self.controller_addr = load_json_file(self.controller_addr_location).get("controller_addr")
            except Exception as e:
                logger.warning(f"Failed to load controller_addr file: {e}")
                self.controller_addr = None
        else:
            self.controller_addr = self.controller_addr_location

        if not self.controller_addr or not isinstance(self.controller_addr, str):
            logger.warning("controller_addr is empty or invalid; online tools disabled.")
            return

        if not self.controller_addr.startswith(("http://", "https://")):
            logger.warning(f"Invalid controller_addr: {self.controller_addr}; online tools disabled.")
            return

        try:
            with self.disable_proxy():
                ret = requests.post(
                    self.controller_addr + "/refresh_all_workers",
                    timeout=10,
                )
                ret.raise_for_status()

                ret = requests.post(
                    self.controller_addr + "/list_models",
                    timeout=10,
                )
                ret.raise_for_status()

                models = ret.json().get("models", [])
                if not isinstance(models, list):
                    models = []

                logger.info(f"Online Tools: {models}")
                self.available_online_tools = models

        except Exception as e:
            self.online_tools_init_error = str(e)
            logger.warning(f"Failed to initialize online tools from controller: {e}")
            self.available_online_tools = []

        miss_tool = []
        for tool in EXPECTED_ONLINE_TOOLS:
            if tool not in self.available_online_tools:
                miss_tool.append(tool)

        if len(miss_tool) == 0:
            logger.info("All tools are called successfully")
        else:
            logger.info(f"Not all tools are called successfully, missing tool {miss_tool}")

    def init_offline_tools(self,):
        self.available_offline_tools = list(offline_tool_workers.keys())
        logger.info(f"Offline Tools: {self.available_offline_tools}")
        
    def init_online_tool_addr_dict(self):
        self.online_tool_addr_dict = {}

        if not self.available_online_tools:
            return

        if not self.controller_addr or not str(self.controller_addr).startswith(("http://", "https://")):
            logger.warning("Invalid controller_addr; skip online tool address initialization.")
            return

        for model_name in self.available_online_tools:
            try:
                with self.disable_proxy():
                    ret = requests.post(
                        self.controller_addr + "/get_worker_address",
                        json={"model": model_name},
                        timeout=10,
                    )
                ret.raise_for_status()

                worker_addr = ret.json().get("address", "")
                if not worker_addr:
                    logger.error(f"worker_addr for {model_name} is empty")
                    continue

                self.online_tool_addr_dict[model_name] = worker_addr

            except Exception as e:
                logger.warning(f"Failed to get worker address for {model_name}: {e}")
    
    def _resolve_timeout_sec(self, tool_name, params):
        params = params or {}

        if tool_name in ["AddPoisLayer", "ComputeDistance"]:
            return 180


        if tool_name in ["ChangeDetection", "GetAreaBoundary", "SmallObjectDetection"]:
            return 120

        if tool_name in ["CrossModalInterpret"]:
            return 120

        if tool_name in ["ObjectDetection"]:
            return 600
        return 60

    def call_tool(self,tool_name,params):

        timeout_sec = self._resolve_timeout_sec(tool_name, params)
        ret_message = {"text": f"Failed to call tool {tool_name} for unknown reason, ", "error_code": 1}
        try:
            signal.alarm(timeout_sec)
            if tool_name in self.available_offline_tools:
                try:
                    tool_generate_fn = get_tool_generate_fn(tool_name)
                    if tool_generate_fn is None:
                        ret_message = {"text": f"Tool {tool_name} not found.", "error_code": 1}
                    else:
                        ret_message = tool_generate_fn(params)
                except Exception as e:
                    logger.error(f"Failed to call tool {tool_name}: {e}")
                    ret_message =  {"text": f"Failed to call tool {tool_name}: {e}", "error_code": 1}
                
            elif tool_name in self.available_online_tools:
                try:
                    tool_worker_addr = self.online_tool_addr_dict[tool_name]
                    with self.disable_proxy():
                        ret = requests.post(tool_worker_addr + "/worker_generate",headers=self.headers,json=params)
                    ret_message = ret.json()
                except Exception as e:
                    logger.error(f"Failed to call tool {tool_name}: {e}")
                    ret_message = {"text": f"Failed to call tool {tool_name}: {e}", "error_code": 1}
            else:
                ret_message = {"text": f"Tool {tool_name} not found.", "error_code": 1}
            signal.alarm(0)
        except TimeoutException as te:
            logger.error(f"Timeout calling tool {tool_name}: {te}")
            ret_message = {"text": f"Timeout calling tool {tool_name}: {te}", "error_code": 1}
        finally:
            signal.alarm(0)
            return ret_message
            
    @contextmanager
    def disable_proxy(self):
        # 保存代理环境变量
        old_HTTP = os.environ.get("HTTP_PROXY")
        old_HTTPS = os.environ.get("HTTPS_PROXY")
        old_http = os.environ.get("http_proxy")
        old_https = os.environ.get("https_proxy")
        
        # 移除代理设置
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        
        try:
            yield
        finally:
            # 恢复代理设置
            if old_http is not None:
                os.environ["http_proxy"] = old_http
            if old_https is not None:
                os.environ["https_proxy"] = old_https
            if old_HTTP is not None:
                os.environ["HTTP_PROXY"] = old_HTTP
            if old_HTTPS is not None:
                os.environ["HTTPS_PROXY"] = old_HTTPS

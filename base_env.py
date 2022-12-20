# This Base-env simulator is copied from:
# https://github.com/sagar-pa/abr_rl_test/blob/e03d209603cc241910e607015cac9e22684ffab5/base_env.py


from abc import abstractclassmethod
import json
import socket
import struct
from typing import Any, Callable
from pathlib import Path

N_ACTIONS = 10
ACTION_SPACE = [i for i in range(N_ACTIONS)]


class BaseEnv:
    def __init__(self, model_path: str, server_address: str):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_path = Path(server_address)
        if server_path.exists() and server_path.is_socket():
            try:
                self.sock.connect(server_address)
            except:
                print("Unable to connect Python IPC socket.")
                raise RuntimeError
        else:
            print("Invalid Python IPC Path")
            raise ValueError
        self.past_action = None
        self.model = self.setup_env(model_path)

    @abstractclassmethod
    def setup_env(self, model_path: str) -> Callable:
        """
        Sets up the model and environment variables with the path provided.
        The callable takes as input the output of process_env_info, and returns
        an int [0-9], signifying the next action to take.
        Args:
            model_path: The path to the model to load as a string.

        Returns:
            A Callable that predicts the next bitrate to send given the input
        """
        raise NotImplementedError("Setup must be done by implementing class")

    @abstractclassmethod
    def process_env_info(self, env_info: dict) -> Any:
        """
        Processes the current environment information to feed to the model.
        Handles, for example, frame stacking, invalid data, normalization.
        Args:
            env_info: the dictionary passed in by Puffer server

        Returns:
            input to be fed into the model for prediction
        """
        raise NotImplementedError("Processing must be done by implementing class")

    def _recv_env_info(self) -> dict:
        json_len_struct = self.sock.recv(2, socket.MSG_WAITALL)
        json_len, *_ = struct.unpack("!H", json_len_struct)
        json_data = self.sock.recv(json_len, socket.MSG_WAITALL)
        env_info = json.loads(json_data)
        return env_info

    def _send_action(self, action: int) -> None:
        action_json = json.dumps(dict(action=action))
        action_json = action_json.encode("utf-8")
        json_len_struct = struct.pack("!H", len(action_json))
        self.sock.sendall(json_len_struct + action_json)

    def env_loop(self) -> None:
        while True:
            try:
                env_info = self._recv_env_info()
            except Exception as e:
                print("{}: {} {}".format(
                    "Encountered error", e.args, "while receiving env info."))
                raise RuntimeError
            model_input = self.process_env_info(env_info=env_info)
            action = self.model(model_input)
            if action not in ACTION_SPACE:
                print("Action not contained in the action space.")
                raise ValueError
            self.past_action = action
            self._send_action(action)
from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Callable

import requests
import socketio
import torch
from tqdm import tqdm

from ... import CONFIG
from ...logger import logger, remote_logger
from .LocalBackend import LocalBackend, LocalMixin

if TYPE_CHECKING:

    from ...schema.Request import RequestModel


def handle_response(handle_result: Callable, url: str, event: str, data: Any) -> bool:

    from ...schema.Response import ResponseModel, ResultModel

    # Load the data into the ResponseModel pydantic class.
    response = ResponseModel(**data)

    # Log response for user
    remote_logger.info(str(response))

    # If the status of the response is completed, update the local nodes that the user specified to save.
    # Then disconnect and continue.
    if response.status == ResponseModel.JobStatus.COMPLETED:
        # Create BytesIO object to store bytes received from server in.
        result_bytes = io.BytesIO()
        result_bytes.seek(0)

        # Get result from result url using job id.
        with requests.get(
            url=f"http{'s' if CONFIG.API.SSL else ''}://{url}/result/{response.id}",
            stream=True,
        ) as stream:
            # Total size of incoming data.
            total_size = float(stream.headers["Content-length"])

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Downloading result",
            ) as progress_bar:
                # chunk_size=None so server determines chunk size.
                for data in stream.iter_content(chunk_size=None):
                    progress_bar.update(len(data))
                    result_bytes.write(data)

        # Move cursor to beginning of bytes.
        result_bytes.seek(0)

        # Decode bytes with pickle and then into pydantic object.
        result: "ResultModel" = ResultModel(
            **torch.load(result_bytes, map_location="cpu")
        )

        # Close bytes
        result_bytes.close()

        handle_result(result.value)

        return True
    # Or if there was some error.
    elif response.status == ResponseModel.JobStatus.ERROR:
        raise Exception(str(response))

    return False


def blocking_request(url: str, request: "RequestModel", handle_result: Callable):

    from ...schema.Response import ResponseModel

    # Create a socketio connection to the server.
    with socketio.SimpleClient(logger=logger, reconnection_attempts=10) as sio:
        # Connect
        sio.connect(
            f"ws{'s' if CONFIG.API.SSL else ''}://{url}",
            socketio_path="/ws/socket.io",
            transports=["websocket"],
            wait_timeout=10,
        )

        # Give request session ID so server knows to respond via websockets to us.
        request.session_id = sio.sid

        # Submit request via
        response = requests.post(
            f"http{'s' if CONFIG.API.SSL else ''}://{url}/request",
            json=request.model_dump(exclude=["id", "received"]),
            headers={"ndif-api-key": CONFIG.API.APIKEY},
        )

        if response.status_code == 200:

            response = ResponseModel(**response.json())

        else:

            raise Exception(response.reason)

        remote_logger.info(response)

        while True:
            if handle_response(handle_result, url, *sio.receive()):
                break


class RemoteMixin(LocalMixin):
    """To be inherited by objects that want to be able to be executed by the RemoteBackend."""

    def remote_backend_get_model_key(self) -> str:
        """Should return the model_key used to specify which model to run on the remote service.

        Returns:
            str: Model key.
        """

        raise NotImplementedError()

    def remote_backend_postprocess_result(self, local_result: Any) -> Any:
        """Should handle postprocessing the result from local_backend_execute.

        For example moving tensors to cpu/detaching/etc.

        Args:
            local_result (Any): Local execution result.

        Returns:
            Any: Post processed local execution result.
        """

        raise NotImplementedError()

    def remote_backend_handle_result_value(self, value: Any) -> None:
        """Should handle postprocessed result from remote_backend_postprocess_result on return from remote service.

        Args:
            value (Any): Result.
        """

        raise NotImplementedError()


class RemoteBackend(LocalBackend):
    """Backend to execute a context object via a remote service.

    Context object must inherit from RemoteMixin and implement its methods.

    Attributes:

        url (str): Remote host url. Defaults to that set in CONFIG.API.HOST.
    """

    def __init__(self, url: str = None) -> None:

        self.url = url or CONFIG.API.HOST

    def __call__(self, obj: RemoteMixin):

        # Get model key.
        model_key = obj.remote_backend_get_model_key()

        from ...schema.Request import RequestModel

        # Create request using pydantic to parse the object itself.
        request = RequestModel(object=obj, model_key=model_key)

        # Do blocking request.
        blocking_request(self.url, request, obj.remote_backend_handle_result_value)

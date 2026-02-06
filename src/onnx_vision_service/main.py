"""Main entrypoint to run the ONNX Vision Service module."""

import asyncio

from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.services.vision import Vision

from src.onnx_vision_service.onnx_vision_service import OnnxVisionService


async def main():
    """Register the ONNX Vision Service and start the module."""
    Registry.register_resource_creator(
        Vision.API,
        OnnxVisionService.MODEL,
        ResourceCreatorRegistration(
            OnnxVisionService.new_service,
            OnnxVisionService.validate_config,
        ),
    )
    module = Module.from_args()
    module.add_model_from_registry(Vision.API, OnnxVisionService.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())

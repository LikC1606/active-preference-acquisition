"""Static offline Docker provider for native Harbor on the isolated KVM daemon."""
from typing import override
import os

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode


class StaticOfflineDocker(DockerEnvironment):
    @staticmethod
    @override
    def _requires_egress_control(startup_network_policy, phase_network_policies):
        policies = [startup_network_policy, *phase_network_policies]
        if any(policy.network_mode != NetworkMode.NO_NETWORK for policy in policies):
            raise ValueError("The static provider supports no-network phases only")
        return False

    @property
    @override
    def capabilities(self):
        return EnvironmentCapabilities(
            disable_internet=True, mounted=True, docker_compose=True,
        )

    @property
    @override
    def _docker_compose_paths(self):
        return [*super()._docker_compose_paths,
                self._DOCKER_COMPOSE_BUILD_PATH.parent / "docker-compose-no-network.yaml"]

    @override
    async def _apply_network_policy(self, network_policy):
        if network_policy.network_mode != NetworkMode.NO_NETWORK:
            raise ValueError("Static offline containers cannot enable network access")

    @override
    async def exec(self, command, **kwargs):
        if kwargs.get("user") is None:
            kwargs["user"] = f"{os.getuid()}:{os.getgid()}"
        result = await super().exec(command, **kwargs)
        if result.return_code:
            self.logger.error("Container command failed (%s): %s; stdout=%s; stderr=%s",
                              result.return_code, command, (result.stdout or "")[-4096:],
                              (result.stderr or "")[-4096:])
        return result

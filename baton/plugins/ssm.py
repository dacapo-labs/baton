"""Baton SSM Plugin - AWS Systems Manager integration for VPC resource access.

Provides:
- SSM-managed instance discovery
- VPC resource audit (what can be reached via SSM)
- Port forwarding command generation
- Zone-aware AWS account handling
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Singleton instance
_ssm_manager: SSMManager | None = None


@dataclass
class SSMInstance:
    """An EC2 instance with SSM agent."""

    instance_id: str
    name: str | None = None
    platform: str | None = None  # Linux, Windows
    ip_address: str | None = None
    private_ip: str | None = None
    vpc_id: str | None = None
    subnet_id: str | None = None
    availability_zone: str | None = None
    agent_version: str | None = None
    ping_status: str | None = None  # Online, Offline
    last_ping: str | None = None
    iam_role: str | None = None
    security_groups: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "platform": self.platform,
            "ip_address": self.ip_address,
            "private_ip": self.private_ip,
            "vpc_id": self.vpc_id,
            "subnet_id": self.subnet_id,
            "availability_zone": self.availability_zone,
            "agent_version": self.agent_version,
            "ping_status": self.ping_status,
            "last_ping": self.last_ping,
            "iam_role": self.iam_role,
            "security_groups": self.security_groups,
            "tags": self.tags,
        }


@dataclass
class VPCResource:
    """A resource in a VPC that can be reached via SSM port forwarding."""

    resource_type: str  # rds, elasticache, elb, ec2, opensearch, etc.
    resource_id: str
    name: str | None = None
    endpoint: str | None = None
    port: int | None = None
    vpc_id: str | None = None
    subnet_ids: list[str] = field(default_factory=list)
    security_groups: list[str] = field(default_factory=list)
    engine: str | None = None  # postgres, mysql, redis, etc.
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "port": self.port,
            "vpc_id": self.vpc_id,
            "subnet_ids": self.subnet_ids,
            "security_groups": self.security_groups,
            "engine": self.engine,
            "status": self.status,
        }


@dataclass
class SSMManagerConfig:
    """Configuration for SSM manager."""

    default_region: str = "us-east-1"
    cache_ttl: int = 300  # Cache instance list for 5 minutes
    auto_discover_resources: bool = True


class SSMManager:
    """Manage SSM sessions and VPC resource discovery."""

    def __init__(self, config: SSMManagerConfig):
        self.config = config
        self._instance_cache: dict[str, list[SSMInstance]] = {}
        self._resource_cache: dict[str, list[VPCResource]] = {}
        self._cache_time: dict[str, float] = {}

    def _run_aws_cli(
        self,
        service: str,
        command: str,
        args: list[str] | None = None,
        region: str | None = None,
        profile: str | None = None,
    ) -> dict | list | None:
        """Run AWS CLI command and return parsed JSON output."""
        cmd = ["aws", service, command]

        if args:
            cmd.extend(args)

        if region:
            cmd.extend(["--region", region])
        elif self.config.default_region:
            cmd.extend(["--region", self.config.default_region])

        if profile:
            cmd.extend(["--profile", profile])

        cmd.extend(["--output", "json"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                log.warning(f"AWS CLI error: {result.stderr}")
                return None
            return json.loads(result.stdout) if result.stdout else None
        except subprocess.TimeoutExpired:
            log.warning(f"AWS CLI timeout: {' '.join(cmd)}")
            return None
        except json.JSONDecodeError as e:
            log.warning(f"AWS CLI JSON parse error: {e}")
            return None
        except Exception as e:
            log.warning(f"AWS CLI error: {e}")
            return None

    async def get_ssm_instances(
        self,
        region: str | None = None,
        profile: str | None = None,
        force_refresh: bool = False,
    ) -> list[SSMInstance]:
        """Get all SSM-managed instances."""
        cache_key = f"{profile or 'default'}:{region or self.config.default_region}"

        import time
        if not force_refresh and cache_key in self._instance_cache:
            if time.time() - self._cache_time.get(cache_key, 0) < self.config.cache_ttl:
                return self._instance_cache[cache_key]

        # Get SSM instance information
        ssm_data = await asyncio.to_thread(
            self._run_aws_cli,
            "ssm",
            "describe-instance-information",
            None,
            region,
            profile,
        )

        if not ssm_data:
            return []

        instances = []
        instance_ids = []

        for info in ssm_data.get("InstanceInformationList", []):
            instance_id = info.get("InstanceId", "")
            instance_ids.append(instance_id)

            instance = SSMInstance(
                instance_id=instance_id,
                name=info.get("Name") or info.get("ComputerName"),
                platform=info.get("PlatformType"),
                ip_address=info.get("IPAddress"),
                agent_version=info.get("AgentVersion"),
                ping_status=info.get("PingStatus"),
                last_ping=info.get("LastPingDateTime"),
                iam_role=info.get("IamRole"),
            )
            instances.append(instance)

        # Enrich with EC2 data if we have instances
        if instance_ids:
            ec2_data = await asyncio.to_thread(
                self._run_aws_cli,
                "ec2",
                "describe-instances",
                ["--instance-ids"] + instance_ids,
                region,
                profile,
            )

            if ec2_data:
                ec2_map = {}
                for reservation in ec2_data.get("Reservations", []):
                    for ec2 in reservation.get("Instances", []):
                        ec2_map[ec2.get("InstanceId")] = ec2

                for instance in instances:
                    ec2 = ec2_map.get(instance.instance_id, {})
                    instance.private_ip = ec2.get("PrivateIpAddress")
                    instance.vpc_id = ec2.get("VpcId")
                    instance.subnet_id = ec2.get("SubnetId")
                    instance.availability_zone = ec2.get("Placement", {}).get("AvailabilityZone")
                    instance.security_groups = [
                        sg.get("GroupId") for sg in ec2.get("SecurityGroups", [])
                    ]
                    # Get Name tag
                    for tag in ec2.get("Tags", []):
                        instance.tags[tag.get("Key", "")] = tag.get("Value", "")
                        if tag.get("Key") == "Name":
                            instance.name = tag.get("Value")

        self._instance_cache[cache_key] = instances
        self._cache_time[cache_key] = time.time()

        return instances

    async def get_vpc_resources(
        self,
        vpc_id: str | None = None,
        region: str | None = None,
        profile: str | None = None,
        resource_types: list[str] | None = None,
    ) -> list[VPCResource]:
        """Discover VPC resources that can be reached via SSM port forwarding."""
        resources = []

        # Default resource types to discover
        if resource_types is None:
            resource_types = ["rds", "elasticache", "elb", "opensearch", "ec2"]

        # RDS instances and clusters
        if "rds" in resource_types:
            rds_resources = await self._discover_rds(vpc_id, region, profile)
            resources.extend(rds_resources)

        # ElastiCache clusters
        if "elasticache" in resource_types:
            cache_resources = await self._discover_elasticache(vpc_id, region, profile)
            resources.extend(cache_resources)

        # Load balancers (internal)
        if "elb" in resource_types:
            elb_resources = await self._discover_elb(vpc_id, region, profile)
            resources.extend(elb_resources)

        # OpenSearch domains
        if "opensearch" in resource_types:
            os_resources = await self._discover_opensearch(vpc_id, region, profile)
            resources.extend(os_resources)

        # EC2 instances (non-SSM, reachable via SSM jump)
        if "ec2" in resource_types:
            ec2_resources = await self._discover_ec2(vpc_id, region, profile)
            resources.extend(ec2_resources)

        return resources

    async def _discover_rds(
        self, vpc_id: str | None, region: str | None, profile: str | None
    ) -> list[VPCResource]:
        """Discover RDS instances and clusters."""
        resources = []

        # RDS instances
        rds_data = await asyncio.to_thread(
            self._run_aws_cli, "rds", "describe-db-instances", None, region, profile
        )

        if rds_data:
            for db in rds_data.get("DBInstances", []):
                db_vpc = db.get("DBSubnetGroup", {}).get("VpcId")
                if vpc_id and db_vpc != vpc_id:
                    continue

                endpoint = db.get("Endpoint", {})
                resources.append(VPCResource(
                    resource_type="rds",
                    resource_id=db.get("DBInstanceIdentifier", ""),
                    name=db.get("DBInstanceIdentifier"),
                    endpoint=endpoint.get("Address"),
                    port=endpoint.get("Port"),
                    vpc_id=db_vpc,
                    subnet_ids=[
                        s.get("SubnetIdentifier")
                        for s in db.get("DBSubnetGroup", {}).get("Subnets", [])
                    ],
                    security_groups=[
                        sg.get("VpcSecurityGroupId")
                        for sg in db.get("VpcSecurityGroups", [])
                    ],
                    engine=db.get("Engine"),
                    status=db.get("DBInstanceStatus"),
                ))

        # Aurora clusters
        cluster_data = await asyncio.to_thread(
            self._run_aws_cli, "rds", "describe-db-clusters", None, region, profile
        )

        if cluster_data:
            for cluster in cluster_data.get("DBClusters", []):
                # Skip if we already have instances from this cluster
                if vpc_id:
                    # Would need to check VPC - clusters don't directly expose it
                    pass

                resources.append(VPCResource(
                    resource_type="rds-cluster",
                    resource_id=cluster.get("DBClusterIdentifier", ""),
                    name=cluster.get("DBClusterIdentifier"),
                    endpoint=cluster.get("Endpoint"),
                    port=cluster.get("Port"),
                    security_groups=[
                        sg.get("VpcSecurityGroupId")
                        for sg in cluster.get("VpcSecurityGroups", [])
                    ],
                    engine=cluster.get("Engine"),
                    status=cluster.get("Status"),
                ))

        return resources

    async def _discover_elasticache(
        self, vpc_id: str | None, region: str | None, profile: str | None
    ) -> list[VPCResource]:
        """Discover ElastiCache clusters."""
        resources = []

        cache_data = await asyncio.to_thread(
            self._run_aws_cli, "elasticache", "describe-cache-clusters",
            ["--show-cache-node-info"], region, profile
        )

        if cache_data:
            for cluster in cache_data.get("CacheClusters", []):
                nodes = cluster.get("CacheNodes", [])
                endpoint = None
                port = None

                if nodes:
                    ep = nodes[0].get("Endpoint", {})
                    endpoint = ep.get("Address")
                    port = ep.get("Port")

                # Get config endpoint for Redis cluster mode
                config_ep = cluster.get("ConfigurationEndpoint", {})
                if config_ep:
                    endpoint = config_ep.get("Address")
                    port = config_ep.get("Port")

                resources.append(VPCResource(
                    resource_type="elasticache",
                    resource_id=cluster.get("CacheClusterId", ""),
                    name=cluster.get("CacheClusterId"),
                    endpoint=endpoint,
                    port=port,
                    engine=cluster.get("Engine"),
                    status=cluster.get("CacheClusterStatus"),
                    security_groups=cluster.get("SecurityGroups", []),
                ))

        return resources

    async def _discover_elb(
        self, vpc_id: str | None, region: str | None, profile: str | None
    ) -> list[VPCResource]:
        """Discover internal load balancers."""
        resources = []

        # ALB/NLB
        elbv2_data = await asyncio.to_thread(
            self._run_aws_cli, "elbv2", "describe-load-balancers", None, region, profile
        )

        if elbv2_data:
            for lb in elbv2_data.get("LoadBalancers", []):
                # Only internal load balancers
                if lb.get("Scheme") != "internal":
                    continue

                lb_vpc = lb.get("VpcId")
                if vpc_id and lb_vpc != vpc_id:
                    continue

                resources.append(VPCResource(
                    resource_type="elb",
                    resource_id=lb.get("LoadBalancerArn", "").split("/")[-1],
                    name=lb.get("LoadBalancerName"),
                    endpoint=lb.get("DNSName"),
                    port=443,  # Default, would need listener check
                    vpc_id=lb_vpc,
                    security_groups=lb.get("SecurityGroups", []),
                    status=lb.get("State", {}).get("Code"),
                ))

        return resources

    async def _discover_opensearch(
        self, vpc_id: str | None, region: str | None, profile: str | None
    ) -> list[VPCResource]:
        """Discover OpenSearch domains."""
        resources = []

        # List domains first
        list_data = await asyncio.to_thread(
            self._run_aws_cli, "opensearch", "list-domain-names", None, region, profile
        )

        if not list_data:
            return resources

        for domain_info in list_data.get("DomainNames", []):
            domain_name = domain_info.get("DomainName")
            if not domain_name:
                continue

            # Get domain details
            domain_data = await asyncio.to_thread(
                self._run_aws_cli, "opensearch", "describe-domain",
                ["--domain-name", domain_name], region, profile
            )

            if domain_data:
                domain = domain_data.get("DomainStatus", {})
                vpc_options = domain.get("VPCOptions", {})

                # Skip if not in VPC or wrong VPC
                if not vpc_options:
                    continue
                domain_vpc = vpc_options.get("VPCId")
                if vpc_id and domain_vpc != vpc_id:
                    continue

                endpoints = domain.get("Endpoints", {})
                endpoint = endpoints.get("vpc") or domain.get("Endpoint")

                resources.append(VPCResource(
                    resource_type="opensearch",
                    resource_id=domain.get("DomainId", ""),
                    name=domain_name,
                    endpoint=endpoint,
                    port=443,
                    vpc_id=domain_vpc,
                    subnet_ids=vpc_options.get("SubnetIds", []),
                    security_groups=vpc_options.get("SecurityGroupIds", []),
                    engine=f"OpenSearch {domain.get('EngineVersion', '')}",
                ))

        return resources

    async def _discover_ec2(
        self, vpc_id: str | None, region: str | None, profile: str | None
    ) -> list[VPCResource]:
        """Discover EC2 instances (including those without SSM)."""
        resources = []

        filters = [{"Name": "instance-state-name", "Values": ["running"]}]
        if vpc_id:
            filters.append({"Name": "vpc-id", "Values": [vpc_id]})

        args = []
        for f in filters:
            args.extend(["--filters", f"Name={f['Name']},Values={','.join(f['Values'])}"])

        ec2_data = await asyncio.to_thread(
            self._run_aws_cli, "ec2", "describe-instances", args, region, profile
        )

        if ec2_data:
            # Get SSM instances to filter them out
            ssm_instances = await self.get_ssm_instances(region, profile)
            ssm_ids = {i.instance_id for i in ssm_instances}

            for reservation in ec2_data.get("Reservations", []):
                for ec2 in reservation.get("Instances", []):
                    instance_id = ec2.get("InstanceId")

                    # Skip SSM-managed instances (they're in the SSM list)
                    if instance_id in ssm_ids:
                        continue

                    name = None
                    for tag in ec2.get("Tags", []):
                        if tag.get("Key") == "Name":
                            name = tag.get("Value")
                            break

                    resources.append(VPCResource(
                        resource_type="ec2",
                        resource_id=instance_id,
                        name=name,
                        endpoint=ec2.get("PrivateIpAddress"),
                        port=22,  # Assume SSH
                        vpc_id=ec2.get("VpcId"),
                        subnet_ids=[ec2.get("SubnetId")] if ec2.get("SubnetId") else [],
                        security_groups=[
                            sg.get("GroupId") for sg in ec2.get("SecurityGroups", [])
                        ],
                        status=ec2.get("State", {}).get("Name"),
                    ))

        return resources

    def generate_port_forward_command(
        self,
        instance_id: str,
        remote_host: str,
        remote_port: int,
        local_port: int | None = None,
        region: str | None = None,
        profile: str | None = None,
    ) -> dict[str, str]:
        """Generate AWS CLI command for port forwarding."""
        local_port = local_port or remote_port

        cmd_parts = [
            "aws", "ssm", "start-session",
            "--target", instance_id,
            "--document-name", "AWS-StartPortForwardingSessionToRemoteHost",
            "--parameters", json.dumps({
                "host": [remote_host],
                "portNumber": [str(remote_port)],
                "localPortNumber": [str(local_port)],
            }),
        ]

        if region:
            cmd_parts.extend(["--region", region])
        if profile:
            cmd_parts.extend(["--profile", profile])

        return {
            "command": " ".join(cmd_parts),
            "local_port": local_port,
            "remote_host": remote_host,
            "remote_port": remote_port,
            "instance_id": instance_id,
            "connect_string": f"localhost:{local_port}",
        }

    def generate_session_command(
        self,
        instance_id: str,
        region: str | None = None,
        profile: str | None = None,
    ) -> dict[str, str]:
        """Generate AWS CLI command for interactive session."""
        cmd_parts = ["aws", "ssm", "start-session", "--target", instance_id]

        if region:
            cmd_parts.extend(["--region", region])
        if profile:
            cmd_parts.extend(["--profile", profile])

        return {
            "command": " ".join(cmd_parts),
            "instance_id": instance_id,
        }

    async def check_ssm_prerequisites(
        self,
        region: str | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        """Check if SSM prerequisites are met."""
        checks = {
            "aws_cli": False,
            "ssm_plugin": False,
            "aws_credentials": False,
            "ssm_permissions": False,
        }

        # Check AWS CLI
        try:
            result = subprocess.run(
                ["aws", "--version"], capture_output=True, text=True, timeout=5
            )
            checks["aws_cli"] = result.returncode == 0
        except Exception:
            pass

        # Check SSM plugin
        try:
            result = subprocess.run(
                ["session-manager-plugin", "--version"],
                capture_output=True, text=True, timeout=5
            )
            checks["ssm_plugin"] = result.returncode == 0
        except Exception:
            pass

        # Check AWS credentials
        sts_data = await asyncio.to_thread(
            self._run_aws_cli, "sts", "get-caller-identity", None, region, profile
        )
        if sts_data:
            checks["aws_credentials"] = True
            checks["aws_account"] = sts_data.get("Account")
            checks["aws_arn"] = sts_data.get("Arn")

        # Check SSM permissions (try to list instances)
        if checks["aws_credentials"]:
            ssm_data = await asyncio.to_thread(
                self._run_aws_cli, "ssm", "describe-instance-information",
                ["--max-results", "1"], region, profile
            )
            checks["ssm_permissions"] = ssm_data is not None

        checks["ready"] = all([
            checks["aws_cli"],
            checks["ssm_plugin"],
            checks["aws_credentials"],
            checks["ssm_permissions"],
        ])

        return checks

    def get_summary(self) -> dict[str, Any]:
        """Get summary of cached data."""
        return {
            "cached_regions": list(self._instance_cache.keys()),
            "instance_count": sum(len(v) for v in self._instance_cache.values()),
            "resource_count": sum(len(v) for v in self._resource_cache.values()),
        }


def init_ssm_manager(config: SSMManagerConfig | None = None) -> SSMManager:
    """Initialize the SSM manager singleton."""
    global _ssm_manager
    if config is None:
        config = SSMManagerConfig()
    _ssm_manager = SSMManager(config)
    return _ssm_manager


def get_ssm_manager() -> SSMManager | None:
    """Get the SSM manager singleton."""
    return _ssm_manager

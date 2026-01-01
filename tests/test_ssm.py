"""Tests for SSM plugin."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baton.plugins.ssm import (
    SSMInstance,
    SSMManager,
    SSMManagerConfig,
    VPCResource,
    get_ssm_manager,
    init_ssm_manager,
)


class TestSSMInstance:
    """Tests for SSMInstance dataclass."""

    def test_basic_instance(self):
        """Test basic SSM instance creation."""
        instance = SSMInstance(
            instance_id="i-1234567890abcdef0",
            name="web-server-1",
            platform="Linux",
            private_ip="10.0.1.100",
            vpc_id="vpc-12345",
            ping_status="Online",
        )

        assert instance.instance_id == "i-1234567890abcdef0"
        assert instance.name == "web-server-1"
        assert instance.platform == "Linux"
        assert instance.ping_status == "Online"
        assert instance.tags == {}
        assert instance.security_groups == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        instance = SSMInstance(
            instance_id="i-abc123",
            name="test-instance",
            platform="Windows",
            ip_address="1.2.3.4",
            private_ip="10.0.0.5",
            vpc_id="vpc-test",
            subnet_id="subnet-123",
            availability_zone="us-east-1a",
            agent_version="3.1.0",
            ping_status="Online",
            security_groups=["sg-123", "sg-456"],
            tags={"Environment": "test"},
        )

        result = instance.to_dict()

        assert result["instance_id"] == "i-abc123"
        assert result["name"] == "test-instance"
        assert result["platform"] == "Windows"
        assert result["vpc_id"] == "vpc-test"
        assert result["security_groups"] == ["sg-123", "sg-456"]
        assert result["tags"] == {"Environment": "test"}

    def test_default_values(self):
        """Test that default values are properly set."""
        instance = SSMInstance(instance_id="i-minimal")

        assert instance.name is None
        assert instance.platform is None
        assert instance.vpc_id is None
        assert instance.ping_status is None
        assert instance.tags == {}
        assert instance.security_groups == []


class TestVPCResource:
    """Tests for VPCResource dataclass."""

    def test_rds_resource(self):
        """Test RDS resource creation."""
        resource = VPCResource(
            resource_type="rds",
            resource_id="mydb",
            name="mydb",
            endpoint="mydb.abc123.us-east-1.rds.amazonaws.com",
            port=5432,
            vpc_id="vpc-12345",
            engine="postgres",
            status="available",
        )

        assert resource.resource_type == "rds"
        assert resource.port == 5432
        assert resource.engine == "postgres"

    def test_elasticache_resource(self):
        """Test ElastiCache resource creation."""
        resource = VPCResource(
            resource_type="elasticache",
            resource_id="redis-cluster",
            endpoint="redis-cluster.abc123.cache.amazonaws.com",
            port=6379,
            engine="redis",
        )

        assert resource.resource_type == "elasticache"
        assert resource.port == 6379
        assert resource.engine == "redis"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        resource = VPCResource(
            resource_type="elb",
            resource_id="internal-lb",
            name="internal-api",
            endpoint="internal-lb-123.elb.amazonaws.com",
            port=443,
            vpc_id="vpc-test",
            security_groups=["sg-789"],
            status="active",
        )

        result = resource.to_dict()

        assert result["resource_type"] == "elb"
        assert result["resource_id"] == "internal-lb"
        assert result["endpoint"] == "internal-lb-123.elb.amazonaws.com"
        assert result["security_groups"] == ["sg-789"]


class TestSSMManagerConfig:
    """Tests for SSMManagerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SSMManagerConfig()

        assert config.default_region == "us-east-1"
        assert config.cache_ttl == 300
        assert config.auto_discover_resources is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SSMManagerConfig(
            default_region="us-west-2",
            cache_ttl=600,
            auto_discover_resources=False,
        )

        assert config.default_region == "us-west-2"
        assert config.cache_ttl == 600
        assert config.auto_discover_resources is False


class TestSSMManager:
    """Tests for SSMManager."""

    @pytest.fixture
    def manager(self):
        """Create SSM manager instance."""
        config = SSMManagerConfig()
        return SSMManager(config)

    def test_init(self, manager):
        """Test manager initialization."""
        assert manager._instance_cache == {}
        assert manager._resource_cache == {}
        assert manager.config.default_region == "us-east-1"

    def test_generate_port_forward_command(self, manager):
        """Test port forward command generation."""
        result = manager.generate_port_forward_command(
            instance_id="i-abc123",
            remote_host="mydb.rds.amazonaws.com",
            remote_port=5432,
            local_port=15432,
            region="us-east-1",
            profile="dev",
        )

        assert "aws ssm start-session" in result["command"]
        assert "--target i-abc123" in result["command"]
        assert "AWS-StartPortForwardingSessionToRemoteHost" in result["command"]
        assert "--region us-east-1" in result["command"]
        assert "--profile dev" in result["command"]
        assert result["local_port"] == 15432
        assert result["remote_port"] == 5432
        assert result["connect_string"] == "localhost:15432"

    def test_generate_port_forward_command_default_local_port(self, manager):
        """Test port forward uses remote port as local when not specified."""
        result = manager.generate_port_forward_command(
            instance_id="i-abc123",
            remote_host="redis.cache.amazonaws.com",
            remote_port=6379,
        )

        assert result["local_port"] == 6379
        assert result["remote_port"] == 6379

    def test_generate_session_command(self, manager):
        """Test session command generation."""
        result = manager.generate_session_command(
            instance_id="i-xyz789",
            region="us-west-2",
            profile="prod",
        )

        assert result["command"] == "aws ssm start-session --target i-xyz789 --region us-west-2 --profile prod"
        assert result["instance_id"] == "i-xyz789"

    def test_generate_session_command_minimal(self, manager):
        """Test session command with minimal params."""
        result = manager.generate_session_command(instance_id="i-simple")

        assert result["command"] == "aws ssm start-session --target i-simple"

    def test_get_summary_empty(self, manager):
        """Test get_summary with empty cache."""
        summary = manager.get_summary()

        assert summary["cached_regions"] == []
        assert summary["instance_count"] == 0
        assert summary["resource_count"] == 0

    def test_get_summary_with_data(self, manager):
        """Test get_summary with cached data."""
        manager._instance_cache["default:us-east-1"] = [
            SSMInstance(instance_id="i-1"),
            SSMInstance(instance_id="i-2"),
        ]
        manager._resource_cache["vpc-123"] = [
            VPCResource(resource_type="rds", resource_id="db1"),
        ]

        summary = manager.get_summary()

        assert "default:us-east-1" in summary["cached_regions"]
        assert summary["instance_count"] == 2
        assert summary["resource_count"] == 1


class TestSSMManagerAWSCalls:
    """Tests for SSM Manager AWS CLI interactions."""

    @pytest.fixture
    def manager(self):
        """Create SSM manager instance."""
        config = SSMManagerConfig()
        return SSMManager(config)

    def test_run_aws_cli_success(self, manager):
        """Test successful AWS CLI call."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"test": "data"}'

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = manager._run_aws_cli("sts", "get-caller-identity")

            assert result == {"test": "data"}
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "aws"
            assert call_args[1] == "sts"
            assert call_args[2] == "get-caller-identity"

    def test_run_aws_cli_with_profile_and_region(self, manager):
        """Test AWS CLI call with profile and region."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{}'

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            manager._run_aws_cli(
                "ec2",
                "describe-instances",
                region="us-west-2",
                profile="myprofile",
            )

            call_args = mock_run.call_args[0][0]
            assert "--region" in call_args
            assert "us-west-2" in call_args
            assert "--profile" in call_args
            assert "myprofile" in call_args

    def test_run_aws_cli_failure(self, manager):
        """Test AWS CLI call failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Access Denied"

        with patch("subprocess.run", return_value=mock_result):
            result = manager._run_aws_cli("ssm", "describe-instance-information")

            assert result is None

    def test_run_aws_cli_timeout(self, manager):
        """Test AWS CLI call timeout."""
        with patch("subprocess.run", side_effect=TimeoutError()):
            result = manager._run_aws_cli("ssm", "describe-instance-information")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_ssm_instances_empty(self, manager):
        """Test get_ssm_instances with no instances."""
        with patch.object(manager, "_run_aws_cli", return_value=None):
            result = await manager.get_ssm_instances()

            assert result == []

    @pytest.mark.asyncio
    async def test_get_ssm_instances_success(self, manager):
        """Test get_ssm_instances with data."""
        ssm_response = {
            "InstanceInformationList": [
                {
                    "InstanceId": "i-abc123",
                    "Name": "web-1",
                    "PlatformType": "Linux",
                    "IPAddress": "10.0.1.5",
                    "AgentVersion": "3.1.0",
                    "PingStatus": "Online",
                }
            ]
        }

        ec2_response = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-abc123",
                            "PrivateIpAddress": "10.0.1.5",
                            "VpcId": "vpc-123",
                            "SubnetId": "subnet-456",
                            "Placement": {"AvailabilityZone": "us-east-1a"},
                            "SecurityGroups": [{"GroupId": "sg-789"}],
                            "Tags": [{"Key": "Name", "Value": "web-server-1"}],
                        }
                    ]
                }
            ]
        }

        call_count = 0
        def mock_aws_cli(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if args[0] == "ssm":
                return ssm_response
            elif args[0] == "ec2":
                return ec2_response
            return None

        with patch.object(manager, "_run_aws_cli", side_effect=mock_aws_cli):
            result = await manager.get_ssm_instances()

            assert len(result) == 1
            assert result[0].instance_id == "i-abc123"
            assert result[0].name == "web-server-1"  # From EC2 tags
            assert result[0].vpc_id == "vpc-123"
            assert result[0].security_groups == ["sg-789"]

    @pytest.mark.asyncio
    async def test_get_ssm_instances_caching(self, manager):
        """Test that instances are cached."""
        ssm_response = {
            "InstanceInformationList": [
                {"InstanceId": "i-cached", "PingStatus": "Online"}
            ]
        }

        call_count = 0
        def mock_aws_cli(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if args[0] == "ssm":
                return ssm_response
            return {"Reservations": []}

        with patch.object(manager, "_run_aws_cli", side_effect=mock_aws_cli):
            # First call
            result1 = await manager.get_ssm_instances()
            # Second call should use cache
            result2 = await manager.get_ssm_instances()

            assert len(result1) == 1
            assert len(result2) == 1
            # AWS CLI should only be called twice (ssm + ec2) for first request
            assert call_count == 2


class TestSSMManagerResourceDiscovery:
    """Tests for VPC resource discovery."""

    @pytest.fixture
    def manager(self):
        """Create SSM manager instance."""
        config = SSMManagerConfig()
        return SSMManager(config)

    @pytest.mark.asyncio
    async def test_discover_rds(self, manager):
        """Test RDS discovery."""
        rds_response = {
            "DBInstances": [
                {
                    "DBInstanceIdentifier": "mydb",
                    "Endpoint": {
                        "Address": "mydb.abc.us-east-1.rds.amazonaws.com",
                        "Port": 5432,
                    },
                    "DBSubnetGroup": {"VpcId": "vpc-123"},
                    "Engine": "postgres",
                    "DBInstanceStatus": "available",
                    "VpcSecurityGroups": [{"VpcSecurityGroupId": "sg-db"}],
                }
            ]
        }

        def mock_aws_cli(service, command, *args, **kwargs):
            if service == "rds" and command == "describe-db-instances":
                return rds_response
            return {"DBClusters": []}

        with patch.object(manager, "_run_aws_cli", side_effect=mock_aws_cli):
            result = await manager._discover_rds(None, None, None)

            assert len(result) == 1
            assert result[0].resource_type == "rds"
            assert result[0].resource_id == "mydb"
            assert result[0].port == 5432
            assert result[0].engine == "postgres"

    @pytest.mark.asyncio
    async def test_discover_elasticache(self, manager):
        """Test ElastiCache discovery."""
        cache_response = {
            "CacheClusters": [
                {
                    "CacheClusterId": "redis-prod",
                    "CacheNodes": [
                        {
                            "Endpoint": {
                                "Address": "redis-prod.cache.amazonaws.com",
                                "Port": 6379,
                            }
                        }
                    ],
                    "Engine": "redis",
                    "CacheClusterStatus": "available",
                }
            ]
        }

        with patch.object(manager, "_run_aws_cli", return_value=cache_response):
            result = await manager._discover_elasticache(None, None, None)

            assert len(result) == 1
            assert result[0].resource_type == "elasticache"
            assert result[0].engine == "redis"
            assert result[0].port == 6379

    @pytest.mark.asyncio
    async def test_discover_elb_internal_only(self, manager):
        """Test ELB discovery only returns internal LBs."""
        elb_response = {
            "LoadBalancers": [
                {
                    "LoadBalancerArn": "arn:aws:elasticloadbalancing:...:loadbalancer/app/internal-lb/abc",
                    "LoadBalancerName": "internal-lb",
                    "Scheme": "internal",
                    "VpcId": "vpc-123",
                    "DNSName": "internal-lb.elb.amazonaws.com",
                    "State": {"Code": "active"},
                },
                {
                    "LoadBalancerArn": "arn:aws:elasticloadbalancing:...:loadbalancer/app/public-lb/xyz",
                    "LoadBalancerName": "public-lb",
                    "Scheme": "internet-facing",
                    "VpcId": "vpc-123",
                    "DNSName": "public-lb.elb.amazonaws.com",
                    "State": {"Code": "active"},
                },
            ]
        }

        with patch.object(manager, "_run_aws_cli", return_value=elb_response):
            result = await manager._discover_elb(None, None, None)

            # Should only return internal LB
            assert len(result) == 1
            assert result[0].name == "internal-lb"

    @pytest.mark.asyncio
    async def test_get_vpc_resources_filters_by_type(self, manager):
        """Test resource discovery respects type filter."""
        with patch.object(manager, "_discover_rds", new_callable=AsyncMock) as mock_rds, \
             patch.object(manager, "_discover_elasticache", new_callable=AsyncMock) as mock_cache, \
             patch.object(manager, "_discover_elb", new_callable=AsyncMock) as mock_elb, \
             patch.object(manager, "_discover_opensearch", new_callable=AsyncMock) as mock_os, \
             patch.object(manager, "_discover_ec2", new_callable=AsyncMock) as mock_ec2:

            mock_rds.return_value = []
            mock_cache.return_value = []
            mock_elb.return_value = []
            mock_os.return_value = []
            mock_ec2.return_value = []

            # Only request RDS and ElastiCache
            await manager.get_vpc_resources(resource_types=["rds", "elasticache"])

            mock_rds.assert_called_once()
            mock_cache.assert_called_once()
            mock_elb.assert_not_called()
            mock_os.assert_not_called()
            mock_ec2.assert_not_called()


class TestSSMPrerequisites:
    """Tests for SSM prerequisites checking."""

    @pytest.fixture
    def manager(self):
        """Create SSM manager instance."""
        config = SSMManagerConfig()
        return SSMManager(config)

    @pytest.mark.asyncio
    async def test_check_prerequisites_all_pass(self, manager):
        """Test prerequisites check when all pass."""
        def mock_subprocess(*args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            return result

        sts_response = {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"}
        ssm_response = {"InstanceInformationList": []}

        def mock_aws_cli(service, command, *args, **kwargs):
            if service == "sts":
                return sts_response
            if service == "ssm":
                return ssm_response
            return None

        with patch("subprocess.run", side_effect=mock_subprocess), \
             patch.object(manager, "_run_aws_cli", side_effect=mock_aws_cli):
            result = await manager.check_ssm_prerequisites()

            assert result["aws_cli"] is True
            assert result["ssm_plugin"] is True
            assert result["aws_credentials"] is True
            assert result["ssm_permissions"] is True
            assert result["ready"] is True
            assert result["aws_account"] == "123456789012"

    @pytest.mark.asyncio
    async def test_check_prerequisites_missing_plugin(self, manager):
        """Test prerequisites check with missing SSM plugin."""
        def mock_subprocess(cmd, *args, **kwargs):
            result = MagicMock()
            if "session-manager-plugin" in cmd:
                raise FileNotFoundError()
            result.returncode = 0
            return result

        sts_response = {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"}

        with patch("subprocess.run", side_effect=mock_subprocess), \
             patch.object(manager, "_run_aws_cli", return_value=sts_response):
            result = await manager.check_ssm_prerequisites()

            assert result["aws_cli"] is True
            assert result["ssm_plugin"] is False
            assert result["ready"] is False


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_init_and_get(self):
        """Test singleton initialization and retrieval."""
        config = SSMManagerConfig(default_region="eu-west-1")

        manager1 = init_ssm_manager(config)
        manager2 = get_ssm_manager()

        assert manager1 is manager2
        assert manager1.config.default_region == "eu-west-1"

    def test_init_with_default_config(self):
        """Test singleton initialization with default config."""
        manager = init_ssm_manager()

        assert manager is not None
        assert manager.config.default_region == "us-east-1"

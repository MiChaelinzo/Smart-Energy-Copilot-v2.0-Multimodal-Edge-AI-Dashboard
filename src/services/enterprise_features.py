"""
Enterprise Features Service for multi-tenant, RBAC, and API gateway functionality.

Provides enterprise-grade features including role-based access control,
multi-tenant architecture, API gateway, and advanced analytics.
"""

import asyncio
import json
import jwt
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Integer, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from src.config.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class UserRole(Enum):
    """User roles for RBAC."""
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"
    ENERGY_MANAGER = "energy_manager"
    FACILITY_MANAGER = "facility_manager"
    ANALYST = "analyst"
    VIEWER = "viewer"
    DEVICE_OPERATOR = "device_operator"


class Permission(Enum):
    """System permissions."""
    READ_ENERGY_DATA = "read_energy_data"
    WRITE_ENERGY_DATA = "write_energy_data"
    MANAGE_DEVICES = "manage_devices"
    CONTROL_DEVICES = "control_devices"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"
    MANAGE_TENANTS = "manage_tenants"
    CONFIGURE_SYSTEM = "configure_system"
    ACCESS_API = "access_api"
    EXPORT_DATA = "export_data"


class TenantModel(Base):
    """Tenant database model."""
    __tablename__ = 'tenants'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    domain = Column(String, unique=True, nullable=False)
    subscription_plan = Column(String, nullable=False)
    max_users = Column(Integer, nullable=False)
    max_devices = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    settings = Column(Text)  # JSON string
    
    users = relationship("UserModel", back_populates="tenant")


class UserModel(Base):
    """User database model."""
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True)
    tenant_id = Column(String, ForeignKey('tenants.id'), nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    permissions = Column(Text)  # JSON string
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    tenant = relationship("TenantModel", back_populates="users")


@dataclass
class Tenant:
    """Tenant representation."""
    id: str
    name: str
    domain: str
    subscription_plan: str
    max_users: int
    max_devices: int
    created_at: datetime
    is_active: bool
    settings: Dict[str, Any]
    current_users: int = 0
    current_devices: int = 0


@dataclass
class User:
    """User representation."""
    id: str
    tenant_id: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    permissions: Set[Permission]
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime


@dataclass
class APIKey:
    """API key representation."""
    key_id: str
    tenant_id: str
    user_id: str
    key_hash: str
    name: str
    permissions: Set[Permission]
    rate_limit: int
    is_active: bool
    expires_at: Optional[datetime]
    created_at: datetime
    last_used: Optional[datetime]


class EnterpriseService:
    """Enterprise features service."""
    
    def __init__(self, database_url: str = "sqlite:///enterprise.db", 
                 redis_url: str = "redis://localhost:6379"):
        self.database_url = database_url
        self.redis_url = redis_url
        self.engine = None
        self.SessionLocal = None
        self.redis_client = None
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.SUPER_ADMIN: set(Permission),
            UserRole.TENANT_ADMIN: {
                Permission.READ_ENERGY_DATA, Permission.WRITE_ENERGY_DATA,
                Permission.MANAGE_DEVICES, Permission.CONTROL_DEVICES,
                Permission.VIEW_ANALYTICS, Permission.MANAGE_USERS,
                Permission.ACCESS_API, Permission.EXPORT_DATA
            },
            UserRole.ENERGY_MANAGER: {
                Permission.READ_ENERGY_DATA, Permission.WRITE_ENERGY_DATA,
                Permission.VIEW_ANALYTICS, Permission.ACCESS_API,
                Permission.EXPORT_DATA
            },
            UserRole.FACILITY_MANAGER: {
                Permission.READ_ENERGY_DATA, Permission.MANAGE_DEVICES,
                Permission.CONTROL_DEVICES, Permission.VIEW_ANALYTICS,
                Permission.ACCESS_API
            },
            UserRole.ANALYST: {
                Permission.READ_ENERGY_DATA, Permission.VIEW_ANALYTICS,
                Permission.ACCESS_API, Permission.EXPORT_DATA
            },
            UserRole.VIEWER: {
                Permission.READ_ENERGY_DATA, Permission.VIEW_ANALYTICS
            },
            UserRole.DEVICE_OPERATOR: {
                Permission.READ_ENERGY_DATA, Permission.CONTROL_DEVICES
            }
        }
        
        self.api_keys: Dict[str, APIKey] = {}
        self.rate_limits: Dict[str, Dict[str, int]] = {}
        
    async def initialize(self) -> bool:
        """Initialize enterprise service."""
        logger.info("Initializing Enterprise Service...")
        
        try:
            # Initialize database
            self.engine = create_engine(self.database_url)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            
            # Initialize Redis
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Load API keys
            await self._load_api_keys()
            
            logger.info("Enterprise Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enterprise Service: {e}")
            return False
    
    # Tenant Management
    async def create_tenant(self, tenant_data: Dict[str, Any]) -> Optional[Tenant]:
        """Create a new tenant."""
        try:
            session = self.SessionLocal()
            
            tenant_model = TenantModel(
                id=tenant_data['id'],
                name=tenant_data['name'],
                domain=tenant_data['domain'],
                subscription_plan=tenant_data['subscription_plan'],
                max_users=tenant_data['max_users'],
                max_devices=tenant_data['max_devices'],
                settings=json.dumps(tenant_data.get('settings', {}))
            )
            
            session.add(tenant_model)
            session.commit()
            
            tenant = Tenant(
                id=tenant_model.id,
                name=tenant_model.name,
                domain=tenant_model.domain,
                subscription_plan=tenant_model.subscription_plan,
                max_users=tenant_model.max_users,
                max_devices=tenant_model.max_devices,
                created_at=tenant_model.created_at,
                is_active=tenant_model.is_active,
                settings=json.loads(tenant_model.settings)
            )
            
            session.close()
            logger.info(f"Created tenant: {tenant.name}")
            return tenant
            
        except Exception as e:
            logger.error(f"Error creating tenant: {e}")
            return None
    
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        try:
            session = self.SessionLocal()
            tenant_model = session.query(TenantModel).filter_by(id=tenant_id).first()
            
            if not tenant_model:
                return None
            
            # Count current users and devices
            current_users = session.query(UserModel).filter_by(
                tenant_id=tenant_id, is_active=True
            ).count()
            
            tenant = Tenant(
                id=tenant_model.id,
                name=tenant_model.name,
                domain=tenant_model.domain,
                subscription_plan=tenant_model.subscription_plan,
                max_users=tenant_model.max_users,
                max_devices=tenant_model.max_devices,
                created_at=tenant_model.created_at,
                is_active=tenant_model.is_active,
                settings=json.loads(tenant_model.settings),
                current_users=current_users,
                current_devices=0  # Would be calculated from device service
            )
            
            session.close()
            return tenant
            
        except Exception as e:
            logger.error(f"Error getting tenant: {e}")
            return None
    
    # User Management
    async def create_user(self, user_data: Dict[str, Any]) -> Optional[User]:
        """Create a new user."""
        try:
            session = self.SessionLocal()
            
            # Check tenant limits
            tenant = await self.get_tenant(user_data['tenant_id'])
            if not tenant or tenant.current_users >= tenant.max_users:
                logger.error("Tenant user limit exceeded")
                return None
            
            # Hash password
            password_hash = hashlib.sha256(user_data['password'].encode()).hexdigest()
            
            # Get role permissions
            role = UserRole(user_data['role'])
            permissions = self.role_permissions.get(role, set())
            
            user_model = UserModel(
                id=user_data['id'],
                tenant_id=user_data['tenant_id'],
                email=user_data['email'],
                password_hash=password_hash,
                first_name=user_data['first_name'],
                last_name=user_data['last_name'],
                role=user_data['role'],
                permissions=json.dumps([p.value for p in permissions])
            )
            
            session.add(user_model)
            session.commit()
            
            user = User(
                id=user_model.id,
                tenant_id=user_model.tenant_id,
                email=user_model.email,
                first_name=user_model.first_name,
                last_name=user_model.last_name,
                role=role,
                permissions=permissions,
                is_active=user_model.is_active,
                last_login=user_model.last_login,
                created_at=user_model.created_at
            )
            
            session.close()
            logger.info(f"Created user: {user.email}")
            return user
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    async def authenticate_user(self, email: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token."""
        try:
            session = self.SessionLocal()
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            user_model = session.query(UserModel).filter_by(
                email=email, password_hash=password_hash, is_active=True
            ).first()
            
            if not user_model:
                return None
            
            # Update last login
            user_model.last_login = datetime.utcnow()
            session.commit()
            
            # Generate JWT token
            payload = {
                'user_id': user_model.id,
                'tenant_id': user_model.tenant_id,
                'role': user_model.role,
                'permissions': json.loads(user_model.permissions),
                'exp': datetime.utcnow() + timedelta(hours=24)
            }
            
            token = jwt.encode(payload, 'secret_key', algorithm='HS256')
            
            session.close()
            logger.info(f"User authenticated: {email}")
            return token
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, 'secret_key', algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    # API Gateway
    async def create_api_key(self, key_data: Dict[str, Any]) -> Optional[APIKey]:
        """Create a new API key."""
        try:
            # Generate API key
            import secrets
            api_key = f"ek_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Get permissions
            permissions = set(Permission(p) for p in key_data.get('permissions', []))
            
            api_key_obj = APIKey(
                key_id=key_data['key_id'],
                tenant_id=key_data['tenant_id'],
                user_id=key_data['user_id'],
                key_hash=key_hash,
                name=key_data['name'],
                permissions=permissions,
                rate_limit=key_data.get('rate_limit', 1000),
                is_active=True,
                expires_at=key_data.get('expires_at'),
                created_at=datetime.utcnow()
            )
            
            # Store in memory and Redis
            self.api_keys[api_key] = api_key_obj
            await self._store_api_key(api_key, api_key_obj)
            
            logger.info(f"Created API key: {key_data['name']}")
            return api_key_obj
            
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            return None
    
    async def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key and check rate limits."""
        try:
            # Check if key exists
            if api_key not in self.api_keys:
                key_obj = await self._load_api_key(api_key)
                if not key_obj:
                    return None
                self.api_keys[api_key] = key_obj
            
            key_obj = self.api_keys[api_key]
            
            # Check if key is active
            if not key_obj.is_active:
                return None
            
            # Check expiration
            if key_obj.expires_at and datetime.utcnow() > key_obj.expires_at:
                return None
            
            # Check rate limit
            if not await self._check_rate_limit(api_key, key_obj.rate_limit):
                return None
            
            # Update last used
            key_obj.last_used = datetime.utcnow()
            
            return key_obj
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None
    
    async def check_permission(self, user_or_key: Any, permission: Permission) -> bool:
        """Check if user or API key has permission."""
        try:
            if isinstance(user_or_key, User):
                return permission in user_or_key.permissions
            elif isinstance(user_or_key, APIKey):
                return permission in user_or_key.permissions
            else:
                return False
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False
    
    # Analytics and Reporting
    async def get_tenant_analytics(self, tenant_id: str, 
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get analytics for a tenant."""
        try:
            analytics = {
                'tenant_id': tenant_id,
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'users': {
                    'total': 0,
                    'active': 0,
                    'new_this_period': 0
                },
                'api_usage': {
                    'total_requests': 0,
                    'requests_by_endpoint': {},
                    'rate_limit_hits': 0
                },
                'energy_data': {
                    'total_consumption': 0,
                    'average_daily': 0,
                    'peak_demand': 0,
                    'cost_savings': 0
                },
                'devices': {
                    'total': 0,
                    'online': 0,
                    'by_type': {}
                }
            }
            
            # Get user analytics
            session = self.SessionLocal()
            total_users = session.query(UserModel).filter_by(tenant_id=tenant_id).count()
            active_users = session.query(UserModel).filter_by(
                tenant_id=tenant_id, is_active=True
            ).count()
            new_users = session.query(UserModel).filter(
                UserModel.tenant_id == tenant_id,
                UserModel.created_at >= start_date,
                UserModel.created_at <= end_date
            ).count()
            
            analytics['users'] = {
                'total': total_users,
                'active': active_users,
                'new_this_period': new_users
            }
            
            session.close()
            
            # Get API usage from Redis
            api_stats = await self._get_api_usage_stats(tenant_id, start_date, end_date)
            analytics['api_usage'] = api_stats
            
            # Mock energy and device data (would integrate with actual services)
            analytics['energy_data'] = {
                'total_consumption': 12500.5,
                'average_daily': 45.2,
                'peak_demand': 8.5,
                'cost_savings': 234.50
            }
            
            analytics['devices'] = {
                'total': 25,
                'online': 23,
                'by_type': {
                    'smart_plugs': 8,
                    'thermostats': 3,
                    'lights': 12,
                    'sensors': 2
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting tenant analytics: {e}")
            return {}
    
    async def export_tenant_data(self, tenant_id: str, 
                               data_types: List[str], format: str = 'json') -> Optional[str]:
        """Export tenant data in specified format."""
        try:
            export_data = {}
            
            if 'users' in data_types:
                session = self.SessionLocal()
                users = session.query(UserModel).filter_by(tenant_id=tenant_id).all()
                export_data['users'] = [
                    {
                        'id': user.id,
                        'email': user.email,
                        'first_name': user.first_name,
                        'last_name': user.last_name,
                        'role': user.role,
                        'created_at': user.created_at.isoformat(),
                        'last_login': user.last_login.isoformat() if user.last_login else None
                    }
                    for user in users
                ]
                session.close()
            
            if 'energy_data' in data_types:
                # Mock energy data export
                export_data['energy_consumption'] = [
                    {
                        'timestamp': '2024-01-01T00:00:00Z',
                        'consumption_kwh': 45.2,
                        'cost_usd': 5.42
                    }
                    # ... more data points
                ]
            
            if format == 'json':
                return json.dumps(export_data, indent=2)
            elif format == 'csv':
                # Convert to CSV format
                return self._convert_to_csv(export_data)
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error exporting tenant data: {e}")
            return None
    
    # Private methods
    async def _load_api_keys(self) -> None:
        """Load API keys from Redis."""
        try:
            if self.redis_client:
                keys = self.redis_client.keys("api_key:*")
                for key in keys:
                    key_data = self.redis_client.hgetall(key)
                    if key_data:
                        api_key = key.split(":")[-1]
                        # Reconstruct APIKey object
                        # Implementation would deserialize from Redis
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
    
    async def _store_api_key(self, api_key: str, key_obj: APIKey) -> None:
        """Store API key in Redis."""
        try:
            if self.redis_client:
                key_data = asdict(key_obj)
                key_data['permissions'] = [p.value for p in key_obj.permissions]
                self.redis_client.hset(f"api_key:{api_key}", mapping=key_data)
        except Exception as e:
            logger.error(f"Error storing API key: {e}")
    
    async def _load_api_key(self, api_key: str) -> Optional[APIKey]:
        """Load API key from Redis."""
        try:
            if self.redis_client:
                key_data = self.redis_client.hgetall(f"api_key:{api_key}")
                if key_data:
                    # Reconstruct APIKey object from Redis data
                    # Implementation would deserialize properly
                    pass
            return None
        except Exception as e:
            logger.error(f"Error loading API key: {e}")
            return None
    
    async def _check_rate_limit(self, api_key: str, limit: int) -> bool:
        """Check API key rate limit."""
        try:
            if not self.redis_client:
                return True
            
            current_minute = datetime.utcnow().strftime("%Y-%m-%d:%H:%M")
            key = f"rate_limit:{api_key}:{current_minute}"
            
            current_count = self.redis_client.get(key)
            if current_count is None:
                self.redis_client.setex(key, 60, 1)
                return True
            
            if int(current_count) >= limit:
                return False
            
            self.redis_client.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True
    
    async def _get_api_usage_stats(self, tenant_id: str, 
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get API usage statistics from Redis."""
        try:
            # Mock implementation - would aggregate from Redis logs
            return {
                'total_requests': 15420,
                'requests_by_endpoint': {
                    '/api/energy/consumption': 8500,
                    '/api/devices/status': 3200,
                    '/api/recommendations': 2100,
                    '/api/forecasts': 1620
                },
                'rate_limit_hits': 23
            }
        except Exception as e:
            logger.error(f"Error getting API usage stats: {e}")
            return {}
    
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert data to CSV format."""
        # Mock CSV conversion
        return "timestamp,consumption_kwh,cost_usd\n2024-01-01T00:00:00Z,45.2,5.42\n"


# Global enterprise service instance
enterprise_service = EnterpriseService()
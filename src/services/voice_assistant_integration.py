"""
Voice Assistant Integration Service for Alexa, Google Assistant, and Siri.

Provides natural language interface for energy management and smart home control
through voice commands and conversational AI.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from src.services.smart_home_automation import smart_home_service
from src.services.forecasting_service import forecasting_service
# from src.services.recommendation_engine import recommendation_service  # Will be imported when needed
from src.config.logging import get_logger

logger = get_logger(__name__)


class VoiceAssistant(Enum):
    """Supported voice assistants."""
    ALEXA = "alexa"
    GOOGLE = "google"
    SIRI = "siri"
    CUSTOM = "custom"


class IntentType(Enum):
    """Voice command intent types."""
    ENERGY_STATUS = "energy_status"
    DEVICE_CONTROL = "device_control"
    SCENE_ACTIVATION = "scene_activation"
    FORECAST_REQUEST = "forecast_request"
    RECOMMENDATIONS = "recommendations"
    COST_INQUIRY = "cost_inquiry"
    CONSUMPTION_HISTORY = "consumption_history"
    DEVICE_STATUS = "device_status"
    AUTOMATION_CONTROL = "automation_control"
    HELP = "help"


@dataclass
class VoiceCommand:
    """Voice command representation."""
    command_id: str
    assistant: VoiceAssistant
    raw_text: str
    intent: IntentType
    entities: Dict[str, Any]
    confidence: float
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class VoiceResponse:
    """Voice response representation."""
    response_id: str
    text: str
    speech_text: str
    card_title: Optional[str] = None
    card_content: Optional[str] = None
    should_end_session: bool = True
    reprompt_text: Optional[str] = None


class VoiceAssistantService:
    """Advanced voice assistant integration service."""
    
    def __init__(self):
        self.intent_handlers = {
            IntentType.ENERGY_STATUS: self._handle_energy_status,
            IntentType.DEVICE_CONTROL: self._handle_device_control,
            IntentType.SCENE_ACTIVATION: self._handle_scene_activation,
            IntentType.FORECAST_REQUEST: self._handle_forecast_request,
            IntentType.RECOMMENDATIONS: self._handle_recommendations,
            IntentType.COST_INQUIRY: self._handle_cost_inquiry,
            IntentType.CONSUMPTION_HISTORY: self._handle_consumption_history,
            IntentType.DEVICE_STATUS: self._handle_device_status,
            IntentType.AUTOMATION_CONTROL: self._handle_automation_control,
            IntentType.HELP: self._handle_help
        }
        
        self.intent_patterns = {
            IntentType.ENERGY_STATUS: [
                r"what.*(energy|power|consumption)",
                r"how much.*(energy|electricity|power)",
                r"(energy|power) (status|usage|consumption)"
            ],
            IntentType.DEVICE_CONTROL: [
                r"turn (on|off) (the )?(.+)",
                r"(dim|brighten) (the )?(.+)",
                r"set (.+) to (\d+)",
                r"(start|stop) (the )?(.+)"
            ],
            IntentType.SCENE_ACTIVATION: [
                r"activate (.+) scene",
                r"set (.+) mode",
                r"switch to (.+)",
                r"enable (.+) scene"
            ],
            IntentType.FORECAST_REQUEST: [
                r"(forecast|predict).*(energy|consumption|usage)",
                r"what will.*(energy|power|consumption)",
                r"(tomorrow|next week).*(energy|usage)"
            ],
            IntentType.RECOMMENDATIONS: [
                r"(recommend|suggest|advice).*(energy|save|optimize)",
                r"how (can|to).*(save|reduce).*(energy|power|cost)",
                r"energy (tips|recommendations|suggestions)"
            ],
            IntentType.COST_INQUIRY: [
                r"how much.*(cost|spend|bill)",
                r"what.*(cost|price|bill)",
                r"(monthly|weekly|daily) (cost|bill|expense)"
            ]
        }
        
        self.conversation_context = {}
        self.is_running = False
    
    async def start_service(self) -> bool:
        """Start the voice assistant service."""
        logger.info("Starting Voice Assistant Integration Service...")
        
        try:
            # Initialize voice assistant integrations
            await self._initialize_assistants()
            
            # Start conversation context cleanup
            self.is_running = True
            asyncio.create_task(self._cleanup_contexts())
            
            logger.info("Voice Assistant Service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Voice Assistant Service: {e}")
            return False
    
    async def process_voice_command(self, raw_text: str, assistant: VoiceAssistant,
                                  user_id: str = None, session_id: str = None) -> VoiceResponse:
        """Process a voice command and generate response."""
        logger.info(f"Processing voice command: '{raw_text}' from {assistant.value}")
        
        try:
            # Parse command
            command = await self._parse_voice_command(raw_text, assistant, user_id, session_id)
            
            # Get intent handler
            handler = self.intent_handlers.get(command.intent)
            if not handler:
                return await self._handle_unknown_intent(command)
            
            # Process command
            response = await handler(command)
            
            # Update conversation context
            if session_id:
                self._update_conversation_context(session_id, command, response)
            
            logger.info(f"Generated response for {command.intent.value}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            return VoiceResponse(
                response_id=f"error_{datetime.now().timestamp()}",
                text="I'm sorry, I encountered an error processing your request. Please try again.",
                speech_text="I'm sorry, I encountered an error processing your request. Please try again."
            )
    
    async def register_alexa_skill(self, skill_config: Dict[str, Any]) -> bool:
        """Register Alexa skill configuration."""
        try:
            # Mock Alexa skill registration
            logger.info("Registering Alexa skill...")
            
            # In real implementation, this would:
            # 1. Register with Amazon Developer Console
            # 2. Set up Lambda function endpoints
            # 3. Configure intent schema and utterances
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering Alexa skill: {e}")
            return False
    
    async def register_google_action(self, action_config: Dict[str, Any]) -> bool:
        """Register Google Assistant action."""
        try:
            # Mock Google Action registration
            logger.info("Registering Google Assistant action...")
            
            # In real implementation, this would:
            # 1. Register with Google Actions Console
            # 2. Set up Dialogflow intents
            # 3. Configure fulfillment webhooks
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering Google Action: {e}")
            return False
    
    async def register_siri_shortcut(self, shortcut_config: Dict[str, Any]) -> bool:
        """Register Siri shortcut configuration."""
        try:
            # Mock Siri shortcut registration
            logger.info("Registering Siri shortcuts...")
            
            # In real implementation, this would:
            # 1. Generate Siri shortcut definitions
            # 2. Provide shortcut installation instructions
            # 3. Set up webhook endpoints for shortcut execution
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering Siri shortcuts: {e}")
            return False
    
    async def _parse_voice_command(self, raw_text: str, assistant: VoiceAssistant,
                                 user_id: str = None, session_id: str = None) -> VoiceCommand:
        """Parse raw voice text into structured command."""
        # Normalize text
        normalized_text = raw_text.lower().strip()
        
        # Detect intent
        intent = await self._detect_intent(normalized_text)
        
        # Extract entities
        entities = await self._extract_entities(normalized_text, intent)
        
        # Calculate confidence (mock implementation)
        confidence = 0.85 if intent != IntentType.HELP else 0.95
        
        return VoiceCommand(
            command_id=f"cmd_{datetime.now().timestamp()}",
            assistant=assistant,
            raw_text=raw_text,
            intent=intent,
            entities=entities,
            confidence=confidence,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id
        )
    
    async def _detect_intent(self, text: str) -> IntentType:
        """Detect intent from normalized text."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent
        
        return IntentType.HELP  # Default to help for unknown intents
    
    async def _extract_entities(self, text: str, intent: IntentType) -> Dict[str, Any]:
        """Extract entities from text based on intent."""
        entities = {}
        
        if intent == IntentType.DEVICE_CONTROL:
            # Extract device name and action
            turn_match = re.search(r"turn (on|off) (the )?(.+)", text)
            if turn_match:
                entities['action'] = turn_match.group(1)
                entities['device'] = turn_match.group(3).strip()
            
            dim_match = re.search(r"(dim|brighten) (the )?(.+)", text)
            if dim_match:
                entities['action'] = dim_match.group(1)
                entities['device'] = dim_match.group(3).strip()
            
            set_match = re.search(r"set (.+) to (\d+)", text)
            if set_match:
                entities['device'] = set_match.group(1).strip()
                entities['value'] = int(set_match.group(2))
        
        elif intent == IntentType.SCENE_ACTIVATION:
            # Extract scene name
            scene_match = re.search(r"activate (.+) scene", text)
            if scene_match:
                entities['scene'] = scene_match.group(1).strip()
            
            mode_match = re.search(r"set (.+) mode", text)
            if mode_match:
                entities['scene'] = mode_match.group(1).strip()
        
        elif intent == IntentType.FORECAST_REQUEST:
            # Extract time period
            if "tomorrow" in text:
                entities['period'] = 'tomorrow'
            elif "next week" in text:
                entities['period'] = 'next_week'
            elif "today" in text:
                entities['period'] = 'today'
            else:
                entities['period'] = 'default'
        
        elif intent == IntentType.COST_INQUIRY:
            # Extract time period
            if "monthly" in text:
                entities['period'] = 'month'
            elif "weekly" in text:
                entities['period'] = 'week'
            elif "daily" in text:
                entities['period'] = 'day'
            else:
                entities['period'] = 'current'
        
        return entities
    
    async def _handle_energy_status(self, command: VoiceCommand) -> VoiceResponse:
        """Handle energy status requests."""
        try:
            # Get current energy status (mock data)
            current_consumption = 2.5  # kW
            daily_consumption = 45.2  # kWh
            monthly_consumption = 1250  # kWh
            
            text = f"Your current energy consumption is {current_consumption} kilowatts. " \
                   f"Today you've used {daily_consumption} kilowatt hours, " \
                   f"and this month you've used {monthly_consumption} kilowatt hours."
            
            return VoiceResponse(
                response_id=f"energy_status_{datetime.now().timestamp()}",
                text=text,
                speech_text=text,
                card_title="Energy Status",
                card_content=f"Current: {current_consumption} kW\nToday: {daily_consumption} kWh\nMonth: {monthly_consumption} kWh"
            )
            
        except Exception as e:
            logger.error(f"Error handling energy status: {e}")
            return self._error_response("I couldn't retrieve your energy status right now.")
    
    async def _handle_device_control(self, command: VoiceCommand) -> VoiceResponse:
        """Handle device control commands."""
        try:
            device_name = command.entities.get('device', 'unknown device')
            action = command.entities.get('action', 'control')
            value = command.entities.get('value')
            
            # Find matching device
            devices = await smart_home_service.get_device_status()
            
            # Mock device control
            success = True  # In real implementation, call smart_home_service.control_device()
            
            if success:
                if action in ['on', 'off']:
                    text = f"I've turned {action} the {device_name}."
                elif action in ['dim', 'brighten']:
                    text = f"I've {action}ed the {device_name}."
                elif value:
                    text = f"I've set the {device_name} to {value}."
                else:
                    text = f"I've controlled the {device_name}."
            else:
                text = f"I couldn't control the {device_name}. Please check if it's available."
            
            return VoiceResponse(
                response_id=f"device_control_{datetime.now().timestamp()}",
                text=text,
                speech_text=text,
                card_title="Device Control",
                card_content=f"Device: {device_name}\nAction: {action}"
            )
            
        except Exception as e:
            logger.error(f"Error handling device control: {e}")
            return self._error_response("I couldn't control that device right now.")
    
    async def _handle_scene_activation(self, command: VoiceCommand) -> VoiceResponse:
        """Handle scene activation commands."""
        try:
            scene_name = command.entities.get('scene', 'unknown scene')
            
            # Mock scene activation
            success = True  # In real implementation, call smart_home_service.activate_scene()
            
            if success:
                text = f"I've activated the {scene_name} scene."
            else:
                text = f"I couldn't find or activate the {scene_name} scene."
            
            return VoiceResponse(
                response_id=f"scene_activation_{datetime.now().timestamp()}",
                text=text,
                speech_text=text,
                card_title="Scene Activation",
                card_content=f"Scene: {scene_name}"
            )
            
        except Exception as e:
            logger.error(f"Error handling scene activation: {e}")
            return self._error_response("I couldn't activate that scene right now.")
    
    async def _handle_forecast_request(self, command: VoiceCommand) -> VoiceResponse:
        """Handle energy forecast requests."""
        try:
            period = command.entities.get('period', 'default')
            
            # Mock forecast data
            if period == 'tomorrow':
                forecast_consumption = 48.5
                forecast_cost = 5.82
                text = f"Tomorrow, I predict you'll use about {forecast_consumption} kilowatt hours, " \
                       f"costing approximately ${forecast_cost:.2f}."
            elif period == 'next_week':
                forecast_consumption = 340
                forecast_cost = 40.80
                text = f"Next week, I predict you'll use about {forecast_consumption} kilowatt hours, " \
                       f"costing approximately ${forecast_cost:.2f}."
            else:
                forecast_consumption = 52.1
                forecast_cost = 6.25
                text = f"For the next 24 hours, I predict you'll use about {forecast_consumption} kilowatt hours, " \
                       f"costing approximately ${forecast_cost:.2f}."
            
            return VoiceResponse(
                response_id=f"forecast_{datetime.now().timestamp()}",
                text=text,
                speech_text=text,
                card_title="Energy Forecast",
                card_content=f"Period: {period}\nConsumption: {forecast_consumption} kWh\nCost: ${forecast_cost:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error handling forecast request: {e}")
            return self._error_response("I couldn't generate an energy forecast right now.")
    
    async def _handle_recommendations(self, command: VoiceCommand) -> VoiceResponse:
        """Handle energy recommendation requests."""
        try:
            # Mock recommendations
            recommendations = [
                "Lower your thermostat by 1 degree to save $15 monthly",
                "Use LED bulbs to reduce lighting costs by 30%",
                "Unplug devices when not in use to eliminate standby power"
            ]
            
            text = "Here are my top energy saving recommendations: " + ". ".join(recommendations[:2])
            
            return VoiceResponse(
                response_id=f"recommendations_{datetime.now().timestamp()}",
                text=text,
                speech_text=text,
                card_title="Energy Recommendations",
                card_content="\n".join(f"• {rec}" for rec in recommendations)
            )
            
        except Exception as e:
            logger.error(f"Error handling recommendations: {e}")
            return self._error_response("I couldn't generate recommendations right now.")
    
    async def _handle_cost_inquiry(self, command: VoiceCommand) -> VoiceResponse:
        """Handle cost inquiry requests."""
        try:
            period = command.entities.get('period', 'current')
            
            # Mock cost data
            if period == 'month':
                cost = 125.50
                text = f"Your monthly energy cost is approximately ${cost:.2f}."
            elif period == 'week':
                cost = 28.75
                text = f"Your weekly energy cost is approximately ${cost:.2f}."
            elif period == 'day':
                cost = 4.12
                text = f"Your daily energy cost is approximately ${cost:.2f}."
            else:
                cost = 4.12
                text = f"Your current daily energy cost is approximately ${cost:.2f}."
            
            return VoiceResponse(
                response_id=f"cost_inquiry_{datetime.now().timestamp()}",
                text=text,
                speech_text=text,
                card_title="Energy Cost",
                card_content=f"Period: {period}\nCost: ${cost:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error handling cost inquiry: {e}")
            return self._error_response("I couldn't retrieve cost information right now.")
    
    async def _handle_consumption_history(self, command: VoiceCommand) -> VoiceResponse:
        """Handle consumption history requests."""
        try:
            # Mock historical data
            text = "Over the past week, your average daily consumption was 45 kilowatt hours. " \
                   "Your highest usage day was Tuesday with 52 kilowatt hours, " \
                   "and your lowest was Sunday with 38 kilowatt hours."
            
            return VoiceResponse(
                response_id=f"history_{datetime.now().timestamp()}",
                text=text,
                speech_text=text,
                card_title="Consumption History",
                card_content="Weekly Average: 45 kWh/day\nHighest: Tuesday (52 kWh)\nLowest: Sunday (38 kWh)"
            )
            
        except Exception as e:
            logger.error(f"Error handling consumption history: {e}")
            return self._error_response("I couldn't retrieve consumption history right now.")
    
    async def _handle_device_status(self, command: VoiceCommand) -> VoiceResponse:
        """Handle device status requests."""
        try:
            # Mock device status
            online_devices = 12
            total_devices = 15
            total_consumption = 2.8
            
            text = f"You have {online_devices} out of {total_devices} devices online, " \
                   f"currently consuming {total_consumption} kilowatts total."
            
            return VoiceResponse(
                response_id=f"device_status_{datetime.now().timestamp()}",
                text=text,
                speech_text=text,
                card_title="Device Status",
                card_content=f"Online: {online_devices}/{total_devices}\nConsumption: {total_consumption} kW"
            )
            
        except Exception as e:
            logger.error(f"Error handling device status: {e}")
            return self._error_response("I couldn't retrieve device status right now.")
    
    async def _handle_automation_control(self, command: VoiceCommand) -> VoiceResponse:
        """Handle automation control requests."""
        try:
            text = "Your automation system is running with 5 active rules. " \
                   "Energy optimization mode is enabled."
            
            return VoiceResponse(
                response_id=f"automation_{datetime.now().timestamp()}",
                text=text,
                speech_text=text,
                card_title="Automation Status",
                card_content="Active Rules: 5\nOptimization: Enabled"
            )
            
        except Exception as e:
            logger.error(f"Error handling automation control: {e}")
            return self._error_response("I couldn't access automation controls right now.")
    
    async def _handle_help(self, command: VoiceCommand) -> VoiceResponse:
        """Handle help requests."""
        text = "I can help you with energy management. You can ask me about your energy status, " \
               "control smart devices, activate scenes, get forecasts, or ask for energy saving recommendations. " \
               "What would you like to know?"
        
        return VoiceResponse(
            response_id=f"help_{datetime.now().timestamp()}",
            text=text,
            speech_text=text,
            card_title="Energy Copilot Help",
            card_content="Available commands:\n• Energy status\n• Device control\n• Scene activation\n• Forecasts\n• Recommendations",
            should_end_session=False,
            reprompt_text="What would you like to know about your energy usage?"
        )
    
    async def _handle_unknown_intent(self, command: VoiceCommand) -> VoiceResponse:
        """Handle unknown intents."""
        text = "I'm not sure I understood that. You can ask me about your energy usage, " \
               "control smart devices, or get energy saving recommendations. What would you like to know?"
        
        return VoiceResponse(
            response_id=f"unknown_{datetime.now().timestamp()}",
            text=text,
            speech_text=text,
            should_end_session=False,
            reprompt_text="What would you like to know about your energy usage?"
        )
    
    def _error_response(self, message: str) -> VoiceResponse:
        """Generate error response."""
        return VoiceResponse(
            response_id=f"error_{datetime.now().timestamp()}",
            text=message,
            speech_text=message
        )
    
    async def _initialize_assistants(self) -> None:
        """Initialize voice assistant integrations."""
        # Mock initialization for each assistant
        logger.info("Initializing voice assistant integrations...")
        
        # Alexa skill setup
        await self.register_alexa_skill({
            'skill_name': 'Energy Copilot',
            'invocation_name': 'energy copilot'
        })
        
        # Google Assistant action setup
        await self.register_google_action({
            'action_name': 'Energy Copilot',
            'invocation_phrase': 'talk to energy copilot'
        })
        
        # Siri shortcuts setup
        await self.register_siri_shortcut({
            'shortcut_name': 'Energy Status',
            'phrase': 'Check my energy usage'
        })
    
    def _update_conversation_context(self, session_id: str, command: VoiceCommand, 
                                   response: VoiceResponse) -> None:
        """Update conversation context for session."""
        if session_id not in self.conversation_context:
            self.conversation_context[session_id] = {
                'commands': [],
                'last_activity': datetime.now()
            }
        
        self.conversation_context[session_id]['commands'].append({
            'command': command,
            'response': response,
            'timestamp': datetime.now()
        })
        self.conversation_context[session_id]['last_activity'] = datetime.now()
    
    async def _cleanup_contexts(self) -> None:
        """Clean up old conversation contexts."""
        while self.is_running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=1)
                expired_sessions = [
                    session_id for session_id, context in self.conversation_context.items()
                    if context['last_activity'] < cutoff_time
                ]
                
                for session_id in expired_sessions:
                    del self.conversation_context[session_id]
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Error cleaning up contexts: {e}")
                await asyncio.sleep(300)


# Global voice assistant service instance
voice_assistant_service = VoiceAssistantService()
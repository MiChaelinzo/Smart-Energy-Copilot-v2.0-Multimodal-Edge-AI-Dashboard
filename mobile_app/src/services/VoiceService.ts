/**
 * Voice Service for mobile app
 * Handles voice recognition and text-to-speech
 */

import Voice from 'react-native-voice';
import Tts from 'react-native-tts';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';

export interface VoiceCommand {
  text: string;
  confidence: number;
  timestamp: Date;
}

export interface VoiceResponse {
  text: string;
  speechText: string;
  success: boolean;
}

class VoiceService {
  private isInitialized: boolean = false;
  private isListening: boolean = false;
  private baseUrl: string = 'http://localhost:8000/api';
  private onResultCallback?: (result: VoiceCommand) => void;
  private onErrorCallback?: (error: string) => void;

  async initialize(): Promise<void> {
    try {
      // Initialize Voice Recognition
      Voice.onSpeechStart = this.onSpeechStart.bind(this);
      Voice.onSpeechEnd = this.onSpeechEnd.bind(this);
      Voice.onSpeechResults = this.onSpeechResults.bind(this);
      Voice.onSpeechError = this.onSpeechError.bind(this);

      // Initialize Text-to-Speech
      await Tts.setDefaultLanguage('en-US');
      await Tts.setDefaultRate(0.5);
      await Tts.setDefaultPitch(1.0);

      // Load backend URL
      const savedUrl = await AsyncStorage.getItem('backend_url');
      if (savedUrl) {
        this.baseUrl = savedUrl;
      }

      this.isInitialized = true;
      console.log('Voice Service initialized');
    } catch (error) {
      console.error('Voice Service initialization error:', error);
    }
  }

  private onSpeechStart(): void {
    console.log('Speech recognition started');
    this.isListening = true;
  }

  private onSpeechEnd(): void {
    console.log('Speech recognition ended');
    this.isListening = false;
  }

  private onSpeechResults(event: any): void {
    const results = event.value;
    if (results && results.length > 0) {
      const command: VoiceCommand = {
        text: results[0],
        confidence: 0.9, // Voice doesn't provide confidence, using default
        timestamp: new Date(),
      };

      console.log('Voice command recognized:', command.text);
      
      if (this.onResultCallback) {
        this.onResultCallback(command);
      }
    }
  }

  private onSpeechError(event: any): void {
    console.error('Speech recognition error:', event.error);
    this.isListening = false;
    
    if (this.onErrorCallback) {
      this.onErrorCallback(event.error?.message || 'Speech recognition error');
    }
  }

  async startListening(
    onResult?: (result: VoiceCommand) => void,
    onError?: (error: string) => void
  ): Promise<void> {
    try {
      if (this.isListening) {
        await this.stopListening();
      }

      this.onResultCallback = onResult;
      this.onErrorCallback = onError;

      await Voice.start('en-US');
    } catch (error) {
      console.error('Error starting voice recognition:', error);
      if (onError) {
        onError('Failed to start voice recognition');
      }
    }
  }

  async stopListening(): Promise<void> {
    try {
      await Voice.stop();
      this.isListening = false;
    } catch (error) {
      console.error('Error stopping voice recognition:', error);
    }
  }

  async speak(text: string): Promise<void> {
    try {
      await Tts.speak(text);
    } catch (error) {
      console.error('Error speaking text:', error);
    }
  }

  async stopSpeaking(): Promise<void> {
    try {
      await Tts.stop();
    } catch (error) {
      console.error('Error stopping speech:', error);
    }
  }

  async processVoiceCommand(command: string): Promise<VoiceResponse> {
    try {
      const response = await axios.post(`${this.baseUrl}/voice/process`, {
        text: command,
        assistant: 'mobile_app',
        user_id: 'mobile_user', // Would be actual user ID in production
      });

      return {
        text: response.data.text,
        speechText: response.data.speech_text || response.data.text,
        success: true,
      };
    } catch (error) {
      console.error('Error processing voice command:', error);
      
      // Fallback to local processing for common commands
      return this.processCommandLocally(command);
    }
  }

  private processCommandLocally(command: string): VoiceResponse {
    const lowerCommand = command.toLowerCase();

    // Energy status commands
    if (lowerCommand.includes('energy') && (lowerCommand.includes('status') || lowerCommand.includes('consumption'))) {
      return {
        text: "I'll show you your current energy status on the dashboard.",
        speechText: "I'll show you your current energy status on the dashboard.",
        success: true,
      };
    }

    // Device control commands
    if (lowerCommand.includes('turn on') || lowerCommand.includes('turn off')) {
      const action = lowerCommand.includes('turn on') ? 'on' : 'off';
      return {
        text: `I'll turn ${action} the device for you.`,
        speechText: `I'll turn ${action} the device for you.`,
        success: true,
      };
    }

    // Forecast commands
    if (lowerCommand.includes('forecast') || lowerCommand.includes('predict')) {
      return {
        text: "Let me show you the energy forecast.",
        speechText: "Let me show you the energy forecast.",
        success: true,
      };
    }

    // Default response
    return {
      text: "I'm sorry, I didn't understand that command. Please try again.",
      speechText: "I'm sorry, I didn't understand that command. Please try again.",
      success: false,
    };
  }

  async getAvailableVoices(): Promise<any[]> {
    try {
      return await Tts.voices();
    } catch (error) {
      console.error('Error getting available voices:', error);
      return [];
    }
  }

  async setVoice(voiceId: string): Promise<void> {
    try {
      await Tts.setDefaultVoice(voiceId);
    } catch (error) {
      console.error('Error setting voice:', error);
    }
  }

  async setSpeechRate(rate: number): Promise<void> {
    try {
      await Tts.setDefaultRate(rate);
    } catch (error) {
      console.error('Error setting speech rate:', error);
    }
  }

  async setSpeechPitch(pitch: number): Promise<void> {
    try {
      await Tts.setDefaultPitch(pitch);
    } catch (error) {
      console.error('Error setting speech pitch:', error);
    }
  }

  isCurrentlyListening(): boolean {
    return this.isListening;
  }

  async destroy(): Promise<void> {
    try {
      await this.stopListening();
      await this.stopSpeaking();
      Voice.destroy();
    } catch (error) {
      console.error('Error destroying voice service:', error);
    }
  }
}

export default new VoiceService();
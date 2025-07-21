import json
import os
from pathlib import Path
from typing import Dict, List, Any

# ConfigManager: Manages client configuration data stored in JSON format.
# Handles loading, saving, validating, and CRUD operations for client settings including keywords and email distribution lists.
class ConfigManager:
    def __init__(self, configPath: str = "ClientConfig.json"):
        self.configPath = configPath
        self.config = self.loadClientConfig()
    
    def loadClientConfig(self) -> Dict[str, Any]:
        """Load client configuration from JSON file"""
        try:
            with open(self.configPath, 'r', encoding='utf-8') as file:
                config = json.load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.configPath} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.configPath}")
    
    def saveClientConfig(self, config: Dict[str, Any]) -> None:
        """Save client configuration to JSON file"""
        try:
            with open(self.configPath, 'w', encoding='utf-8') as file:
                json.dump(config, file, indent=2, ensure_ascii=False)
            self.config = config
        except Exception as e:
            raise Exception(f"Failed to save configuration: {str(e)}")
    
    def getEnabledClients(self) -> List[Dict[str, Any]]:
        """Get list of enabled clients"""
        return [client for client in self.config.get('clients', []) if client.get('enabled', False)]
    
    def getClientByName(self, clientName: str) -> Dict[str, Any]:
        """Get specific client configuration by name"""
        for client in self.config.get('clients', []):
            if client.get('name') == clientName:
                return client
        return None
    
    def addClient(self, clientData: Dict[str, Any]) -> None:
        """Add a new client to configuration"""
        if 'clients' not in self.config:
            self.config['clients'] = []
        
        # Validate required fields
        requiredFields = ['name', 'keywords', 'distribution_emails']
        for field in requiredFields:
            if field not in clientData:
                raise ValueError(f"Missing required field: {field}")
        
        # Check if client already exists
        existingClient = self.getClientByName(clientData['name'])
        if existingClient:
            raise ValueError(f"Client '{clientData['name']}' already exists")
        
        # Set default enabled status
        if 'enabled' not in clientData:
            clientData['enabled'] = True
        
        self.config['clients'].append(clientData)
        self.saveClientConfig(self.config)
    
    def updateClient(self, clientName: str, updatedData: Dict[str, Any]) -> None:
        """Update existing client configuration"""
        for i, client in enumerate(self.config.get('clients', [])):
            if client.get('name') == clientName:
                # Preserve name field
                updatedData['name'] = clientName
                self.config['clients'][i] = updatedData
                self.saveClientConfig(self.config)
                return
        raise ValueError(f"Client '{clientName}' not found")
    
    def removeClient(self, clientName: str) -> None:
        """Remove client from configuration"""
        for i, client in enumerate(self.config.get('clients', [])):
            if client.get('name') == clientName:
                del self.config['clients'][i]
                self.saveClientConfig(self.config)
                return
        raise ValueError(f"Client '{clientName}' not found")
    
    def validateConfig(self) -> bool:
        """Validate configuration structure"""
        if 'clients' not in self.config:
            return False
        
        for client in self.config['clients']:
            requiredFields = ['name', 'keywords', 'distribution_emails']
            for field in requiredFields:
                if field not in client:
                    return False
            
            # Validate email format (basic check)
            for email in client['distribution_emails']:
                if '@' not in email or '.' not in email:
                    return False
        
        return True 
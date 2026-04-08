from typing import Optional, Iterator

from agent.src.agent import Agent
from agent.src.llm import LLM
from agent.src.message import Message

import logging
logger = logging.getLogger(__name__)

class SimpleAgent(Agent):
    """Simple Chat agent"""

    def __init__(
        self, 
        name: str, 
        llm: LLM, 
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, llm, system_prompt, **kwargs)
   
    
    def build_messages(self, input_text: str) -> list[dict]:
        """
        Simple prompt assembly. 
        """
        messages = []
        
        # 1. Add the baseline system instruction
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # 2. Add raw history as-is (Simple implementation)
        for msg in self._history:
            messages.append(msg.to_dict())
            
        # 3. Add current query
        messages.append({"role": "user", "content": input_text})
        
        return messages
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Non-streaming execution. 
        Best for batch processing, automated tools, and internal logic.
        """
        # 1. Assemble the messages
        messages = self.build_messages(input_text)
        
        try:
            # 2. Call the LLM (Synchronous/Blocking)
            response_text = self.llm.generate(messages, **kwargs)
            
            # 3. Transaction Commit: Save to history only on success
            self.add_message(Message(role="user", content=input_text))
            self.add_message(Message(role="assistant", content=response_text))
            
            return response_text
            
        except Exception as e:
            logger.error("Run failed for %s: %s", self.name, e)
            raise
    
    
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        messages = self.build_messages(input_text)
        full_response = ""
        success = False
        
        try:
            for chunk in self.llm.stream(messages, **kwargs):
                full_response += chunk
                yield chunk
            success = True # Only set to True if we finish the whole loop
        
        except GeneratorExit:
            logger.warning(f"User aborted {self.name} stream.")
            raise # Re-raise to close generator correctly

        except Exception as e:
            # We log the error immediately where we have access to 'e'
            logger.error(f"Technical failure in {self.name}: {e}")
            raise # Re-raise to inform the UI
            
        finally:
            if success:
                self.add_message(Message(role="user", content=input_text))
                self.add_message(Message(role="assistant", content=full_response))
                logger.info("Chat turn committed to memory.")
       
        
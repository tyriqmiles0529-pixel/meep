import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from functools import lru_cache

# Path hack for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class MEEPAnalyst:
    """
    The AI Analyst Agent for MEEP.
    Uses LLM tool-calling to interact with the TerminalEngine.
    """
    def __init__(self, engine, api_key: Optional[str] = None):
        self.engine = engine
        
        # Priority: User Key > ENV:GROQ_API_KEY > ENV:OPENAI_API_KEY
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        
        self.api_key = api_key or self.groq_key or self.openai_key
        self.is_groq = bool(self.groq_key and not api_key) or (api_key and "gsk_" in api_key)
        
        if HAS_OPENAI and self.api_key:
            if self.is_groq:
                self.client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
                self.model = "llama-3.3-70b-versatile"
            else:
                self.client = OpenAI(api_key=self.api_key)
                self.model = "gpt-4o"
        else:
            self.client = None
            self.model = None
        
        self.system_prompt = f"""
You are the MEEP AI Analyst (Machine-driven Ensemble Evaluation Platform).
Your goal is to help users understand NBA analytics, model predictions, and betting strategies.

CORE DIRECTIVE:
- ALWAYS prioritize REAL data from the production ledger (`betting_ledger.csv`) when reporting performance.
- NEVER use or mention "mock data" or "placeholders" if real data is available.
- MEEP focuses on 'Survival Mode' (conservative staking) and 'Phase S2' hardening.

ANALYTICS & DRIFT:
- If asked about performance, use `get_ledger_history` to see actual wins, losses, and EV.
- Explain "Model Drift" as the discrepancy between predicted win probabilities and actual outcomes observed in the ledger.
- ALWAYS be professional, data-driven, and transparent.

CURRENT SYSTEM TIME: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        # Multi-level Cache (Short TTL for transient analytics)
        self._tool_cache = {}

    def get_tools(self) -> List[Dict[str, Any]]:
        """Define the tools available to the LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_slate",
                    "description": "Get today's portfolio slate (Core, Growth, Moonshot picks).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "Target date in YYYY-MM-DD format (optional)."}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bankroll_stats",
                    "description": "Get current bankroll status, ROI, and system freshness.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_player_intelligence",
                    "description": "Get correlation data and statistical insights for a specific player.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string", "description": "Full name of the NBA player."}
                        },
                        "required": ["player_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_batch_intelligence",
                    "description": "Get statistical insights for multiple players at once for comparison.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "player_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of full names of the NBA players."
                            }
                        },
                        "required": ["player_names"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_ledger_history",
                    "description": "Get the most recent entries from the production betting ledger (real performance).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Number of recent records to fetch (default 10)."}
                        }
                    }
                }
            }
        ]

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute the engine method corresponding to the tool with local caching."""
        cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        
        # Check cache (5-minute logical TTL check could be added if needed)
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        try:
            result = ""
            if tool_name == "get_slate":
                date_str = args.get("date")
                slate = self.engine.get_portfolio_slate(date_str)
                if not slate: result = "analytics_unavailable: no slate for date"
                else: result = json.dumps({k: v for k, v in slate.items() if k != "raw_picks"}, default=str)
                
            elif tool_name == "get_bankroll_stats":
                stats = self.engine.get_stats()
                if not stats: result = "analytics_unavailable: bankroll data missing"
                else: result = json.dumps(stats, default=str)
                
            elif tool_name == "get_player_intelligence":
                p_name = args.get("player_name")
                corr = self.engine.get_player_correlation(p_name)
                if not corr: result = "analytics_unavailable: player data missing"
                else: result = json.dumps(corr, default=str)

            elif tool_name == "get_batch_intelligence":
                names = args.get("player_names", [])
                batch_res = {}
                for name in names:
                    batch_res[name] = self.engine.get_player_correlation(name) or "analytics_unavailable"
                result = json.dumps(batch_res, default=str)
                
            elif tool_name == "get_ledger_history":
                limit = args.get("limit", 10)
                history = self.engine.get_ledger_history(limit)
                if isinstance(history, dict) and "error" in history: result = f"analytics_unavailable: {history['error']}"
                else: result = json.dumps(history, default=str)
            else:
                result = "analytics_unavailable: tool not implemented"
            
            # Store in cache
            self._tool_cache[cache_key] = result
            return result
        except Exception as e:
            logging.error(f"Tool Error {tool_name}: {e}")
            return "analytics_unavailable: engine error"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Handle the chat loop with pruning and tool calling."""
        if not self.client:
            return "⚠️ AI Analyst Offline: No API Key found. Please set GROQ_API_KEY or OPENAI_API_KEY in your environment."

        # PRUNING: Only send the last 10 messages to keep context focused and tokens low
        pruned_history = messages[-10:] if len(messages) > 10 else messages
        full_messages = [{"role": "system", "content": self.system_prompt}] + pruned_history
        
        try:
            # First LLM call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                tools=self.get_tools(),
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            
            # Handle tool calls
            if msg.tool_calls:
                full_messages.append(msg)
                for tool_call in msg.tool_calls:
                    result = self.call_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
                    full_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": result
                    })
                
                # Second LLM call for final response
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages
                )
                return final_response.choices[0].message.content
                
            return msg.content
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

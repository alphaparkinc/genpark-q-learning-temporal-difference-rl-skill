"""MCP Server for Q-Learning Skill."""
import json
import sys
from client import QLearningAgent

def main():
    agents = {}
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "tools/list":
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [{
                            "name": "train_q_learning",
                            "description": "Train tabular Q-learning agent on transitions",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "states": {"type": "array", "items": {"type": "string"}},
                                    "actions": {"type": "array", "items": {"type": "string"}},
                                    "transitions": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "s": {"type": "string"},
                                                "a": {"type": "string"},
                                                "r": {"type": "number"},
                                                "next_s": {"type": "string"}
                                            },
                                            "required": ["s", "a", "r", "next_s"]
                                        }
                                    }
                                },
                                "required": ["states", "actions", "transitions"]
                            }
                        }]
                    }
                }
            elif method == "tools/call":
                args = params.get("arguments", {})
                agent = QLearningAgent(args["states"], args["actions"])
                for tr in args["transitions"]:
                    agent.update(tr["s"], tr["a"], tr["r"], tr["next_s"])
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps({"policy": agent.get_policy(), "q_table": agent.q_table})}]}
                }
            else:
                res = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
            print(json.dumps(res), flush=True)
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(e)}}
            print(json.dumps(err), flush=True)

if __name__ == "__main__":
    main()

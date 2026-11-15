# An App's replicas each take a machine, and the emulated Domino's proxy spreads requests over them by an assumed rule

To compare topologies on equal hardware (four App replicas of 2 CPUs each, however a topology divides them) the emulation has to run several replicas of one App behind its one address, which ADR 0001 left out. The emulated Domino now starts an App on as many machines as its version's minimum replicas, and its proxy, which every call to an App already passes, chooses the replica that answers each request. We chose this over moving the emulation to Kubernetes because it is a small change to what exists, and because Kubernetes would not have told us the one thing that is missing: Domino documents that requests spread across an App's replicas when session affinity is off, not by which rule.

## Considered options

- **Kubernetes (GKE), as ADR 0001 said to revisit.** It would make three things real: pods with CPU requests and limits on shared nodes, autoscaling that adds and removes replicas under load, and replicas sharing one disk. Rejected for now: it replaces the placement over SSH, the machines' start-up scripts, the model caches and the cost cap (a node pool replaces a deleted node), and the balancing rule would still be ours to pick with the ingress. Revisit when what must be measured is autoscaling itself, or replicas sharing a node.
- **A separate balancer (nginx or HAProxy) on the Workspace.** Rejected: a second program to configure at every deploy, while the proxy already handles identity and paths for every call.
- **One rule only.** Rejected: the rule is the unknown, so both rules Kubernetes itself uses can be chosen, and a comparison can be repeated under the other to see whether the choice moves the numbers.

## Consequences

- Replaces the first consequence of ADR 0001: an App takes one machine per replica, the minimum of its autoscaling; the maximum, the targets and the delays are accepted and ignored, so nothing scales during a run.
- The balancing rule is an assumption until it is checked on a real Domino: publish an App with two replicas that answers its host name, and call it a hundred times with and without keep-alive. The next replica in turn for each request is the default; a replica drawn at random for each caller connection is the other.
- Session affinity is not emulated: the Apps here keep no state per caller.
- Each machine keeps its own dataset folder, where Domino's replicas share one disk: every replica restores or converts its own models at first start.
- A request is sent to every replica an App holds, running or not: a machine lost during a run shows as failed requests, and that run is void.

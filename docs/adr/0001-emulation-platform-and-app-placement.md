# Terraform builds the emulation's platform; Apps are placed on its machines through the fake Domino's Apps API

To rehearse a Domino deployment on Google Cloud, and to compare topologies there, Terraform creates everything that already exists when someone arrives on a real Domino: the network, the fake Domino with its Workspace, the vector store, a pool of machines per hardware tier and a bucket standing in for a Dataset. Apps are never Terraform's: `rag-deploy` publishes them through the fake Domino's Apps API, unchanged, and the fake places each App on a free machine of its hardware tier and runs its entry script there over SSH. We chose this so that the real deploy path (the API calls, the entry scripts, `app.sh`'s own bootstrap) is what gets rehearsed, with the smallest stand-in for Domino's scheduler.

## Considered options

- **Kubernetes (GKE).** The fake turns an App into a Deployment, a Service and an autoscaler. Closest to Domino (pods with CPU limits, real autoscaling), rejected for now: it needs container images, a cluster and a Kubernetes client inside the fake. Revisit if autoscaling or pod CPU limits must be measured.
- **Static.** Terraform starts every App's process and the fake only records what `rag-deploy` sends. Smallest, rejected because `rag-deploy` would not really be rehearsed.
- **`gcloud` scripts in place of Terraform.** Rejected for want of a plan to read before anything is created and of a destroy afterwards.

## Consequences

- One machine per App: an App's `autoscalingSpecification` is accepted and ignored, and an App whose hardware tier has no free machine stays `Queued`.
- A machine is a whole VM: an App sees the machine's CPUs, not a container's quota as on Domino.
- Every call to an App passes the fake's proxy, a single Python process, so each comparison includes one run that bypasses it to show its share of the numbers. Absolute numbers do not transfer to Domino; differences between topologies do.
- The key the fake uses to reach the machines is in Terraform's state, so the bucket holding that state stays private.

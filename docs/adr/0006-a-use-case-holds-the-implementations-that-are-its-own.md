# A use case holds the implementations that are its own, and the toolkit loads them as extensions

ADR 0005 kept every line of code in the toolkit, so that a second use case would be a package of files. The RAG then needed a router that chooses with a kNN model trained on its own utterances, which no other use case would assemble. A use case can therefore hold code. The toolkit holds what any use case can assemble: the ports with their request and answer, the stage types, the API formats and the edge conditions. A use case holds its implementations of them, its trained models and its data. Its workflow file names the packages to load under `extensions`, the toolkit imports every module of each, and their classes register themselves in the toolkit's registries.

## Considered options

- **Settings only**, as ADR 0005 decided. Rejected: the kNN router, its training command and its trained file would live in the toolkit, which would then ship one use case's model.
- **The whole router in the use case**: its port, its request and answer, and its stage. Rejected by the owner: another use case would have to define all of it again to add a router of its own. With the port in the toolkit it writes one class.
- **Python entry points** to find a use case's code, with nothing in the workflow file. Rejected: the file would not say what it loads, and two use cases installed together would load each other's registrations.

## Consequences

- The toolkit still never imports nor names a use case: the package's name is a value of the use case's own workflow file. The structure test of ADR 0005 stays.
- A use case adds registrations and never replaces one of the toolkit's. A name registered twice is refused when the package is loaded.
- A use case imports the toolkit only through `query_processing.extension_api`, and its package follows the toolkit's layers. A structure test of the use case checks both.
- The deploy tool and the model server load the extensions too, since both read the workflow file's components.
- A use case's distribution has packages and dependencies of its own (scikit-learn, for the RAG's router) and can install commands of its own (`qp-rag-train-router`).
- A trained model is trained by a command, serialized, and committed with a manifest of what it was trained on. Start-up refuses a file that its manifest does not match.
- ADR 0005's consequence that a new stage type is always a change to the toolkit becomes a judgment: what any use case can assemble goes to the toolkit.

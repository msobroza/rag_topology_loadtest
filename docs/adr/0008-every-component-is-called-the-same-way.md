# Every component is called the same way, and a user subclasses one abstract class

Adding a component type meant edits in eleven files of six packages, and adding an implementation meant knowing a registry's name, an opener's signature and which mixin comes first. Every component now answers `ainvoke(request)` with an answer, where the request and the answer are domain classes, one pair for each component type. A stage's `reads` fill the request's attributes in order, a key of the stage that names an attribute fills it, and `writes` take the answer's attributes in order. A component type is one abstract class, `BaseComponent[Request, Answer]`. An implementation is one subclass of it that names its backend and its settings block on the class line, builds itself with `from_settings` and writes `invoke`.

## Considered options

- **Mixins the author picks** for each serving mode. Rejected: the author had to know which one to take and that it comes before the port. The mixins are now inside the toolkit's classes.
- **A registry and an opener function for each implementation**, as the toolkit opens its own backends. Rejected by the owner for a user's code: no abstract class tells the registry's name, the decorator's arguments or the opener's signature.
- **A method name for each component type**, such as `choose` for a router. Rejected by the owner: all components share one API. The names `invoke`, `ainvoke` and `from_...` are LangChain's, and so is the default by which `ainvoke` runs `invoke` off the event loop. Nothing subclasses LangChain's `Runnable`.
- **Rewriting the adapters of the embedder, the reranker, the LLM and the vector store** onto `invoke`. Rejected: `ainvoke` is written once in each port's abstract class from its current methods, so no adapter changes, and ingestion and the model server keep those methods.
- **A stage type for each component type.** Rejected: a stage that only hands its fields to its component needs none. `generate` and `retrieve` remain, because they do more than call.
- **The request and the answer on the wire as they are.** Rejected by the owner: a domain class never goes on the wire, an API format translates. The toolkit builds one that mirrors the request and the answer unless a format is registered for the component type.
- **A library's classes in the domain**, such as semantic-router's `Route`. Rejected by the owner: the domain holds our own classes, and a library gives at most the words.

## Consequences

- Every stage states `reads` and `writes` in the workflow file, the two remaining stage types included. ADR 0004's rule that a stage type declares its fields moves to the file.
- The lists bind by position, so the order of a request's and an answer's attributes is part of what a component type promises. Start-up checks the count and the types, and its error shows both orders.
- The class builds itself from its own settings block. This bends the rule that only the composition reads the settings: the composition still finds the class, checks the block against its settings class and gathers its resources.
- `served` exists for every such component type with no code: the toolkit builds the class that calls a model server and the model server's route, on the `fastapi` kind only. A use case writes a `served` class only to redefine it.
- A mistake is refused when the class is defined, naming what to write: no `invoke`, no backend, or a backend named twice. So a use case has one implementation for each backend of a component type.
- Two spellings fill the same registries: a keyword on the class line for a component, a decorator for the toolkit's other entries.
- The toolkit ships a test class for each component type, which a user runs on an implementation.
- The router is the toolkit's fifth component type and takes a query's vector. A router that reads the text would be another component type.

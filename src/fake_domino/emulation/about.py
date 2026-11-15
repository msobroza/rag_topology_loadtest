"""The emulation's part of the Swagger description: the App proxy and placement.

Each says what is documented by Domino, what is assumed, and what belongs to the
emulation only.

Example:
    >>> create_fake_domino(apps, about=EMULATION_ABOUT)
"""

from __future__ import annotations

from .identity import API_KEY_HEADER, FAKE_API_KEY, FAKE_RUN_TOKEN, FAKE_USERNAME

_PLACEMENT_ABOUT = """Placement (`--placement FILE`; emulation only, Domino has \
no such thing): each hardware tier is a pool of machines.

- an App runs on as many machines (replicas) as its version's \
`autoscalingSpecification.minReplicas` when it is `enabled`, else on one; \
`maxReplicas`, the targets, the windows and `useSessionAffinity` are stored, \
never acted on (no scaling during a run);
- a new App takes that many free machines of its tier, all or none, and keeps \
them for the fake's lifetime (the 6.2 API has no stop call); a new version \
starts on the same machines, under a new instance id, taking more or freeing \
the surplus as its replicas ask;
- too few free machines: the instance is `Queued` (a previous version keeps \
running), and each GET of the App tries again;
- the App still has one instance; it reports what its machines run: \
`Running` when all run, `Failed` when one failed, else `Preparing` (a unit is \
up, its port not yet);
- the version's `entryScript` runs over SSH after Domino's run variables \
(`DOMINO_API_PROXY`, `DOMINO_HARDWARE_TIER_ID`, `DOMINO_PROJECT_ID`, \
`DOMINO_APP_ID`, `DOMINO_APP_VERSION_ID`, `DOMINO_RUN_HOST_PATH`) and the \
project environment variables (`--project-env FILE`);
- the App proxy forwards to one of the App's machines unless `--upstream` \
names it: the next in turn for each request, or with `--replica-balancing \
connection-random` one random machine per caller connection;
- the hardware tiers route reports the file's tiers, and an unknown id is 404.

Assumed (Domino's answers are undocumented): an unknown `hardwareTierId` is \
400, as are more replicas than its tier has machines, and a version asking for \
another tier than its App's machines is 409, each as a `FailureEnvelopeV1`; \
with session affinity off, requests spread across the replicas by a rule \
Domino does not state (both of Kubernetes' rules can be chosen; session \
affinity is not emulated); `DOMINO_APP_ID` and `DOMINO_APP_VERSION_ID` are set \
in an App's run; `DOMINO_RUN_HOST_PATH` is the instance's legacy path prefix."""
_PROXY_ABOUT = f"""App proxy (`/apps/{{app_id}}/...` and \
`/{{owner}}/{{project}}/r/notebookSession/{{instance_id}}/...`), forwarding to \
`--upstream NAME=URL` by App name.

Credentials: `Authorization: Bearer {FAKE_RUN_TOKEN}` (what `/access-token` \
answers) or `{API_KEY_HEADER}: {FAKE_API_KEY}`, both the user `{FAKE_USERNAME}`.

Documented by Domino:

- the proxy checks access before a request reaches the App; Agents \
(`AISYSTEM`) share it;
- the App gets `domino-username: <viewer>`, and with secure identity \
propagation `Authorization: Bearer <RS256 JWT>` (audience `apps`; `sub`, \
`preferred_username`, `email`, `given_name`, `family_name`), keys at \
`/auth/realms/DominoRealm/protocol/openid-connect/certs`;
- with `extendedIdentityPropagationToAppsEnabled` the Bearer value is the \
viewer's own token, and no `domino-username` is passed;
- `/access-token` answers a run's token as plain text.

Assumed (not in Domino's docs):

- a script opens an App URL with the API's credentials;
- no valid credential answers 401 (`--unauthenticated redirect`: 302 to the \
login page), before the App is looked up;
- the caller's `Authorization`, `{API_KEY_HEADER}` and `domino-username` are \
removed;
- the Domino prefix is stripped before the App sees the path (`--keep-prefix` \
keeps it);
- the JWT's `iss` is the realm URL; with an API key, the extended "viewer's \
token" is the run token;
- visibility and `accessStatuses` are stored, never enforced."""


EMULATION_ABOUT = "\n\n".join(
    [
        "Emulation (`rag-domino-emulation`): the core's `--behaviour` scripts "
        "are replaced by placement when a placement file is given; without "
        "one, a new instance is Pending on the first GET, then Running.",
        _PROXY_ABOUT,
        _PLACEMENT_ABOUT,
    ]
)

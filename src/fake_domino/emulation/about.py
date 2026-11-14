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

- a new App takes a free machine of its tier and keeps it for the fake's \
lifetime (the 6.2 API has no stop call); a new version starts on the same \
machine, under a new instance id;
- no free machine: the instance is `Queued`, and each GET of the App tries again;
- the instance reports what its machine runs: `Preparing` (its unit is up, its \
port not yet), `Running`, or `Failed`;
- the version's `entryScript` runs over SSH after Domino's run variables \
(`DOMINO_API_PROXY`, `DOMINO_HARDWARE_TIER_ID`, `DOMINO_PROJECT_ID`, \
`DOMINO_APP_ID`, `DOMINO_APP_VERSION_ID`, `DOMINO_RUN_HOST_PATH`) and the \
project environment variables (`--project-env FILE`);
- the App proxy forwards to the App's machine unless `--upstream` names it;
- the hardware tiers route reports the file's tiers, and an unknown id is 404.

Assumed (Domino's answers are undocumented): an unknown `hardwareTierId` is \
400 and a version asking for another tier than its App's machine is 409, both \
as a `FailureEnvelopeV1`; `DOMINO_APP_ID` and `DOMINO_APP_VERSION_ID` are set \
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

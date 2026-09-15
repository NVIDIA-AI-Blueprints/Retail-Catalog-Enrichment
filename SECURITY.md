# Security Policy: Retail Catalog Enrichment

## Reporting a Vulnerability

If you discover a potential security vulnerability, please **do not open a
public issue or pull request describing it**.

- Report it through the [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/) (preferred).
- Email [psirt@nvidia.com](mailto:psirt@nvidia.com). We encourage encrypted
  reports using the [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key).
- Use this repository's **Security** tab and **Report a vulnerability** option
  when private vulnerability reporting is available.

Include the affected version or branch, vulnerability type, reproduction steps,
proof of concept when available, and an assessment of the potential impact.
NVIDIA PSIRT will acknowledge the report, validate its severity, coordinate a
fix, and publish an advisory when appropriate.

## Security Architecture and Context

Retail Catalog Enrichment is a publicly distributed reference blueprint for
evaluating AI-assisted catalog workflows. It is not a managed, multi-tenant
service and is not production-ready by default. The repository includes a web
UI, a FastAPI orchestration API, model-serving containers, and an optional
Milvus/MinIO policy-document stack.

**Repository Exposure Classification:** Public.

Basis: the source is published in the NVIDIA-AI-Blueprints GitHub organization.

**Service Exposure Classification:** External / Regulated (high confidence).

Basis: the blueprint is publicly distributed and provides deployable network
services, although its supplied configuration targets local evaluation rather
than a managed production environment. This contextual label is not a
vulnerability severity rating.

The default Docker Compose configuration binds published ports to the local host
only. Browser requests enter through the Nginx gateway on port 3000; `/api/`
requests are forwarded to the FastAPI backend on the private Compose network.
The backend sends product data and uploaded content to configured NVIDIA NIMs
and, when enabled, the external web-insights provider. Policy PDFs are parsed by
the backend and indexed in the optional Milvus service.

This blueprint does not implement application-level authentication,
authorization, TLS termination, or rate limiting. Direct exposure to an
untrusted network is unsupported. An operator who makes any service remotely
reachable must place it behind controls appropriate to that environment,
including authentication and authorization, TLS, request-size and rate limits,
network filtering, and audit logging. Setting `CATALOG_BIND_ADDRESS=0.0.0.0`
opts out of the local-only default and must only be done with those controls in
place.

### Threat Model

1. **Untrusted file processing:** Image and PDF upload endpoints parse caller-
   supplied files. Malformed or oversized inputs could exploit a parser or
   exhaust memory, CPU, storage, or vector-database capacity.
2. **Prompt injection and model-output integrity:** Product records, brand
   guidance, and uploaded policy documents can contain text that attempts to
   redirect model behavior. Catalog augmentation, brand styling, and policy-
   review flows place dynamic values in bounded JSON data envelopes,
   keep task instructions in the system role, and validate model output before
   using it. These controls reduce but do not eliminate prompt-injection risk;
   model-generated catalog and compliance results still require review before
   consequential use.
3. **Sensitive data disclosure to model providers:** Product records, images,
   manuals, brand instructions, and web-research prompts may be sent to the
   configured model or search endpoints. Operators must not submit data those
   providers are not approved to process.
4. **Credential or infrastructure exposure:** API keys are supplied through the
   environment, while model, Milvus, and MinIO services use network interfaces.
   Publishing those interfaces or leaking environment/configuration data could
   expose credentials, stored policy content, or expensive compute capacity.
5. **Dependency and model supply chain:** The blueprint relies on Python and
   JavaScript packages plus externally supplied container images and model
   artifacts. A compromised dependency or untrusted replacement image could
   execute with the permissions and data available to its container.

### Accepted Risks

- **Unauthenticated resource use and policy deletion:** The FastAPI application
  does not authenticate callers, including callers of compute-intensive
  generation routes and policy upload/deletion routes. This is accepted only
  for local, single-user, or equivalently controlled blueprint evaluation. It
  is not accepted for direct exposure to untrusted networks.
- **No in-application rate limiting:** The blueprint delegates abuse prevention
  to the operator's gateway or network boundary. Remote deployments must add
  appropriate limits before accepting traffic.

### Critical Security Assumptions

- Published services remain bound to localhost, or an operator-provided gateway
  authenticates and authorizes every remote caller before traffic reaches them.
- TLS is terminated by an operator-controlled gateway whenever traffic leaves
  the local host or another explicitly trusted network boundary.
- Users upload only data approved for the configured model, search, and storage
  services; the sample workflow is not a secure store for confidential or
  regulated data.
- The host, Docker daemon, shared `catalog-network`, mounted volumes, and local
  environment files are trusted and accessible only to authorized users.
- Operators supply unique service credentials and apply least privilege before
  adapting the blueprint for a shared or production-like environment.

## Deployment Scope

The supported default is local evaluation on a trusted workstation or an
equivalently isolated host. Production hardening, identity integration,
multi-tenant isolation, security monitoring, backup/retention controls, and
availability guarantees are outside the scope of this blueprint. Downstream
products must perform their own threat analysis and implement controls suited to
their users, data, and deployment environment.

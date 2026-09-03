# File: embed_client_demo.py
from __future__ import annotations

import time

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from rich.console import Console

console = Console()
MAX_WORKERS = 6
embedder = LlamacppEmbedding(max_workers=MAX_WORKERS)


def timed_batch_embed(
    texts: list[str],
    batch_size: int,
) -> tuple[list[list[float]], float]:
    """Run batch embed and return results + wall-clock time in seconds."""
    start = time.perf_counter()
    embeddings = embedder.embed_parallel(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    return embeddings, elapsed


# === Real-world example data ===
# 100 unique support articles / product FAQs (short-medium length)
ORIGINAL_DOCS = [
    "Our return policy allows returns within 30 days of purchase with original packaging.",
    "To reset your password, click 'Forgot Password' on the login page and follow the email instructions.",
    "The Premium plan includes unlimited storage, priority support, and advanced analytics.",
    "We support payments via credit card, PayPal, and bank transfer in most countries.",
    "If your order is delayed, please check tracking or contact support with your order ID.",
    "All devices come with a 1-year warranty covering manufacturing defects.",
    "To cancel subscription, go to Account Settings > Billing > Cancel Subscription.",
    "Our app is available on iOS 15+ and Android 10+ devices.",
    "For bulk orders over 100 units, please contact sales@company.com for custom pricing.",
    "Data is encrypted in transit (TLS 1.3) and at rest (AES-256).",
    "You can export your data anytime from Settings > Privacy > Export Data.",
    "Troubleshooting steps for login issues: clear cache, try incognito, check credentials.",
    "We offer free shipping on orders over $50 in the continental US.",
    "Product X is not compatible with older OS versions prior to 2022.",
    "To request a refund, submit a ticket with proof of purchase and reason.",
    "Our team responds to support tickets within 24 hours on business days.",
    "Two-factor authentication can be enabled under Security Settings to protect your account.",
    "The mobile app requires at least 200MB of free storage space for offline mode.",
    "Enterprise customers can request dedicated SLAs through their account manager.",
    "Billing cycles are monthly by default; annual plans receive a 15% discount.",
    "API rate limits are set to 1000 requests per minute for standard tier accounts.",
    "Dark mode can be toggled in Display Preferences or follows system settings automatically.",
    "Deleted files remain in the Trash folder for 30 days before permanent removal.",
    "Integration with Slack requires admin approval and OAuth configuration.",
    "Firmware updates are pushed automatically when devices are connected to Wi-Fi.",
    "Guest users can view shared documents but cannot edit or download them.",
    "Search supports boolean operators: AND, OR, NOT, and quoted phrases.",
    "Account verification emails may take up to 10 minutes; check spam folders.",
    "The dashboard refreshes every 60 seconds; manual refresh is available via the toolbar.",
    "Multi-currency support is available for invoices in USD, EUR, GBP, and JPY.",
    "Keyboard shortcuts: Ctrl+S saves, Ctrl+Z undoes, Ctrl+F opens search.",
    "Session timeout occurs after 30 minutes of inactivity for security purposes.",
    "Custom domains require DNS CNAME record configuration and SSL validation.",
    "Audit logs retain user activity records for 90 days on Business plans.",
    "Push notifications can be customized per channel in Notification Settings.",
    "File uploads are limited to 500MB per file on Standard plans.",
    "SSO integration supports SAML 2.0 and OIDC protocols for enterprise identity providers.",
    "Maintenance windows are scheduled Sundays 2-4 AM UTC; status page provides updates.",
    "Accessibility features include screen reader support and high contrast mode.",
    "Webhooks can be configured to trigger on user creation, deletion, and plan changes.",
    "Historical reports can be generated for any date range up to 2 years back.",
    "Team workspaces allow role-based access control with Admin, Editor, and Viewer roles.",
    "Browser extensions are available for Chrome, Firefox, Edge, and Safari.",
    "Data residency options include US-East, EU-West, and AP-Southeast regions.",
    "Onboarding sessions are complimentary for teams of 10 or more users.",
    "Password requirements: minimum 12 characters, including uppercase, number, and symbol.",
    "Shared links expire after 7 days unless set to 'No Expiration' by admins.",
    "CSV exports include UTF-8 BOM for compatibility with Excel international characters.",
    "Mobile biometric unlock supports Face ID, Touch ID, and Android fingerprint sensors.",
    "Compliance certifications: SOC 2 Type II, ISO 27001, GDPR, and HIPAA eligible.",
    "Automated backups run daily at midnight UTC and are retained for 30 days.",
    "Video conferencing integrations supported: Zoom, Teams, Meet, and Webex.",
    "Usage analytics show active users, feature adoption, and session duration trends.",
    "Invitation emails are valid for 72 hours; expired invites must be resent.",
    "Custom fields can be added to user profiles via Admin Console > User Schema.",
    "Rate limit exceeded errors return HTTP 429 with Retry-After header guidance.",
    "Localization available in 28 languages; UI language follows browser preference.",
    "Document version history retains last 50 versions with restore capability.",
    "API keys should be rotated every 90 days; old keys are immediately invalidated.",
    "Offline sync conflicts are resolved using last-write-wins strategy by default.",
    "Email digest frequency options: real-time, daily summary, or weekly roundup.",
    "Hardware accessories ship separately and may arrive 3-5 days after main device.",
    "Beta features can be opted into via Labs section in Settings; feedback encouraged.",
    "GDPR data subject requests processed within 30 days per regulatory requirement.",
    "Load balancers distribute traffic across three availability zones for redundancy.",
    "Comment threads support @mentions, emoji reactions, and markdown formatting.",
    "Trial periods last 14 days with full feature access; no credit card required.",
    "DNS propagation for custom domains typically completes within 1-24 hours.",
    "Memory usage optimization tips: close unused tabs, reduce sync frequency.",
    "Incident reports published post-mortem within 5 business days of resolution.",
    "Third-party app marketplace contains 200+ verified integrations.",
    "User provisioning via SCIM automates onboarding/offboarding with IdP sync.",
    "Color-blind friendly palettes available in chart visualization settings.",
    "Maximum concurrent sessions per user: 5 devices simultaneously.",
    "Archive storage costs $0.004/GB/month after exceeding included quota.",
    "Penetration testing conducted quarterly by independent third-party firms.",
    "Calendar sync supports Google Calendar, Outlook, and Apple Calendar natively.",
    "Error messages include correlation IDs for faster support troubleshooting.",
    "Feature flags allow gradual rollout to percentage-based user cohorts.",
    "Print-friendly PDF exports respect page breaks and hide navigation elements.",
    "Network requirements: outbound HTTPS 443, WebSocket 8080, UDP 3478 for media.",
    "User impersonation audit trail logged for compliance and security reviews.",
    "Template library includes 50+ pre-built workflows for common use cases.",
    "Latency SLA: p99 < 200ms for API endpoints in primary region.",
    "Deprecation notices provided 6 months before feature removal with migration guides.",
    "Collaborative editing supports up to 25 simultaneous cursors per document.",
    "Tax exemption certificates uploaded via Billing > Tax Settings for validation.",
    "SDKs available for Python, Node.js, Go, Ruby, Java, and .NET.",
    "Disaster recovery RTO < 4 hours, RPO < 1 hour for critical systems.",
    "Feedback widget collects NPS scores and qualitative comments in-app.",
    "Role inheritance simplifies permission management across nested team structures.",
    "Certificate pinning enforced on mobile apps to prevent MITM attacks.",
    "Scheduled maintenance notifications sent 48 hours in advance via email and banner.",
    "Data anonymization options for analytics to comply with privacy regulations.",
    "CLI tool supports scripting automation with JSON output and exit codes.",
    "White-label options remove branding for agency and reseller partners.",
]

# Use ORIGINAL_DOCS directly since it now contains 100 unique items
SAMPLE_DOCS = ORIGINAL_DOCS


if __name__ == "__main__":
    console.rule("llama.cpp Embedding Server Performance Demo")
    console.print(
        "\n[bold cyan]Scenario:[/bold cyan] Indexing customer support articles "
        "for semantic search in a helpdesk RAG system.\n"
        "Goal: Measure embedding throughput at different dataset sizes.\n"
    )

    console.print(f"[green]Dataset size:[/green] {len(SAMPLE_DOCS)} documents")
    console.print(
        f"[green]Using concurrent version[/green] (max_workers={MAX_WORKERS}) – "
        f"leverages --parallel {MAX_WORKERS} + continuous batching\n"
    )

    embeddings, time_taken = timed_batch_embed(SAMPLE_DOCS, batch_size=2)
    throughput = len(SAMPLE_DOCS) / time_taken if time_taken > 0 else 0

    console.print(
        f"→ Processed {len(SAMPLE_DOCS)} docs in {time_taken:.2f} seconds "
        f"| ~{throughput:.1f} docs/sec\n"
    )

    # Optional: uncomment to also test smaller / larger batches
    # console.print("Testing with batch_size=8...")
    # _, time_b8 = timed_batch_embed(SAMPLE_DOCS, batch_size=8)
    # console.print(f"  → batch_size=8: {time_b8:.2f} seconds\n")

    console.print(
        "[yellow]Tip:[/yellow] Adjust MAX_WORKERS or batch_size to find optimal "
        "throughput for your hardware. All 100 docs are unique real-world examples."
    )

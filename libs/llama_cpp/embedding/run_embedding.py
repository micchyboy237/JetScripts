import os

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding


def main():
    embedder = LlamacppEmbedding(
        model=os.getenv("LLAMA_CPP_EMBED_MODEL"),  # or e5, nomic-embed, etc.
        base_url=os.getenv("LLAMA_CPP_EMBED_URL"),
        use_cache=True,
        use_dynamic_batch_sizing=True,
        verbose=True,
    )

    texts = [
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
    ]

    print("Computing embeddings (single call)...")
    embeddings = embedder.get_embeddings(
        texts,
        return_format="numpy",
        show_progress=True,
    )

    print(f"→ Got {len(embeddings)} embeddings")
    print(f"→ Shape: {embeddings.shape}")
    print(f"→ First vector preview: {embeddings[0][:8]} ...")

    # Also works with a single string
    single_emb = embedder("Hello, world!")
    print(f"Single embedding shape: {single_emb.shape}")


if __name__ == "__main__":
    main()

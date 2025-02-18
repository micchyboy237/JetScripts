from span_marker import SpanMarkerModel

# Load the pre-trained model
model = SpanMarkerModel.from_pretrained(
    "tomaarsen/span-marker-bert-base-cross-ner")

# Example text
text = "Looking for someone to be a part of our agency. Our agency, Vean Global, is a cutting-edge web design and marketing agency specializing in Shopify development, 3D web experiences, and high-performance e-commerce solutions. We\u2019re looking for an experienced Shopify Theme Developer to help us build a fully customizable Shopify theme.\n\n\n\nWhat You'll Be Doing:\n\n\n\n- Developing a high-performance, fully customizable Shopify theme from scratch\n\n- Writing clean, maintainable, and scalable Liquid, JavaScript (Vanilla/React), HTML, and CSS code\n\n- Ensuring theme customization options are user-friendly and intuitive\n\n- Optimizing theme performance for fast loading speeds and smooth UX\n\n- Implementing dynamic content, animations, and advanced customization features\n\n- Troubleshooting and resolving theme-related issues\n\n- Working closely with our team to ensure branding, UI/UX, and functionality align with our goals\n\n\n\nWhat We\u2019re Looking For:\n\n\n\n-  Strong proficiency in Shopify's Liquid templating language\n\n- Expertise in HTML, CSS, JavaScript, and Shopify APIs\n\n- Experience with Shopify metafields and theme customizations\n\n- Strong knowledge of performance optimization best practices\n\n- Experience building custom Shopify themes (portfolio required)\n\n- Ability to work independently and meet deadlines\n\n\n\nLooking forward to working with you!"

# Predict entities
entities = model.predict(text)

print(entities)

from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
myst:
  html_meta:
    "description lang=en": |
      FAQ for AutoGen Studio - A low code tool for building and debugging multi-agent systems
---

# Experimental Features

## Authentication

AutoGen Studio offers an experimental authentication feature to enable personalized experiences (multiple users). Currently, only GitHub authentication is supported. You can extend the base authentication class to add support for other authentication methods.

By default authenticatio is disabled and only enabled when you pass in the `--auth-config` argument when running the application.

### Enable GitHub Authentication

To enable GitHub authentication, create a `auth.yaml` file in your app directory:
"""
logger.info("# Experimental Features")

type: github
jwt_secret: "your-secret-key" # keep secure!
token_expiry_minutes: 60
github:
  client_id: "your-github-client-id"
  client_secret: "your-github-client-secret"
  callback_url: "http://localhost:8081/api/auth/callback"
  scopes: ["user:email"]

"""

"""

**JWT Secret**


- Generate a strong, unique JWT secret (at least 32 random bytes). You can run `openssl rand -hex 32` to generate a secure random key.
- Never commit your JWT secret to version control
- In production, store secrets in environment variables or secure secret management services
- Regularly rotate your JWT secret to limit the impact of potential breaches

**Callback URL**

- The callback URL is the URL that GitHub will redirect to after the user has authenticated. It should match the URL you set in your GitHub OAuth application settings.
- Ensure that the callback URL is accessible from the internet if you are running AutoGen Studio on a remote server.

"""
Please see the documentation on [GitHub OAuth](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authenticating-to-the-rest-api-with-an-oauth-app) for more details on obtaining the `client_id` and `client_secret`.

To pass in this configuration you can use the `--auth-config` argument when running the application:
"""
logger.info("Please see the documentation on [GitHub OAuth](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authenticating-to-the-rest-api-with-an-oauth-app) for more details on obtaining the `client_id` and `client_secret`.")

autogenstudio ui --auth-config /path/to/auth.yaml

"""
Or set the environment variable:
"""
logger.info("Or set the environment variable:")

export AUTOGENSTUDIO_AUTH_CONFIG="/path/to/auth.yaml"

"""

"""

- Authentication is currently experimental and may change in future releases
- User data is stored in your configured database
- When enabled, all API endpoints require authentication except for the authentication endpoints
- WebSocket connections require the token to be passed as a query parameter (`?token=your-jwt-token`)

logger.info("\n\n[DONE]", bright=True)
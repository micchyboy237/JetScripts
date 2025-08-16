from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
- **Contoso Travel** offers luxury vacation packages to exotic destinations worldwide.
- Our premium travel services include personalized itinerary planning and 24/7 concierge support.
- Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage.
- Popular destinations include the Maldives, Swiss Alps, and African safaris.
- Contoso Travel provides exclusive access to boutique hotels and private guided tours.
"""

logger.info("\n\n[DONE]", bright=True)
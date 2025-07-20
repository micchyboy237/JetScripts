import os
import shutil
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.models.model_types import EmbedModelType
from jet.vectors.semantic_search.vector_search_simple import VectorSearch

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Real-world demonstration
if __name__ == "__main__":
    # 1. Specify preffered dimensions
    # dimensions = None
    dimensions = 512
    model_name: EmbedModelType = "mxbai-embed-large"

    search_engine = VectorSearch(model_name, truncate_dim=dimensions)

    # Same sample documents
    sample_docs = [
        "# Fullstack Mobile Developer\nExperience with real-time data synchronization and offline-first mobile app development.\nPreferred Qualifications:\nExperience with TypeScript in React Native development.\nKnowledge of Flutter (as a secondary mobile framework) is a plus.\nFamiliarity with Agile/Scrum development practices.\nPrior experience mentoring junior developers or leading small teams.\nKnowledge of app store optimization (ASO) and mobile analytics.",
        "# Mobile Application Developer - ( Javascript - React Native ) HYBRID SET UP\nBuild Cross-Platform Apps with Upvise, Enhance Web UI with JavaScript, Optimize APIs, and Actively Contribute to Agile Ceremonies\nYOU DESERVE THE BEST - Enjoy These Perks!\nComprehensive Day 1 HMO with 10K medical reimbursement\nAdditional HMO coverage for your family + dental coverage\nFree annual flu vaccine\nContinuous career development and certifications\nTravel & training opportunities overseas\nRegular awards and recognitions\nComprehensive life insurance\n24 Paid Time Offs (with annual leave conversion)\nAnnual merit-based appraisal\nFree daily meals (free breakfast on Mondays & free treats on Fridays)\nRegular engaging company events promoting work-life balance\nAccessible office sites - EASTWOOD (HYBRID)\nEmployee referral programs\nAbout the Role:\nWe're looking for a passionate\nMobile App Developer\nexperienced in JavaScript-based frameworks to help us build smart, scalable mobile and web applications.\nYou'll be part of a collaborative Agile team supporting industry leaders in\nconstruction\n,\nengineering\n, and\nfield services\n.\nWork across a modern stack--including\nUpvise\n(a proprietary JavaScript framework for mobile), and\n.NET Core\nback-end services\nYour Key Responsibilities:\nBuild cross-platform mobile apps using\nUpvise\nEnhance front-end UI using\nJavaScript\n(framework similar to React Native)\nIntegrate and optimize\nAPI\nconnections across our ecosystem\nActively participate in Agile ceremonies like\ndaily stand-ups\n,\nsprint planning\n, and\ncode reviews\nCollaborate with\nProduct, QA, and DevOps\nteams to deliver new features and maintain app performance\nWhat You'll Bring:\n5+ years of commercial experience in a\nSaaS\nenvironment\nProven experience in\nJavaScript\nand\nReact Native\nor similar frameworks\nStrong backend development experience in\n.NET (C#)\nSolid SQL and API integration skills\nUnderstanding of\nSOA\n, scalability, and testability\nA collaborative and Agile mindset\nAmenable to work hybrid - Eastwood Day shift\nTech Stack:\nFrontend:\nUpvise (JavaScript-based framework)\nMobile:\nUpvise Mobile platform (iOS & Android)\nBackend:\n.NET, MVC/Web API, C#, MS SQL Server\nBe part of a company that powers project success across industries.\nBuild smart apps that make an impact.\nApply now and level up your dev career!",
    ]

    search_engine.add_documents(sample_docs)

    # Same example queries
    queries = [
        "React Native",
    ]

    for query in queries:
        results = search_engine.search(query, top_k=len(sample_docs))
        print(f"\nQuery: {query}")
        print("Top matches:")
        for num, (doc, score) in enumerate(results, 1):
            print(f"\n{colorize_log(f"{num}.", "ORANGE")} (Score: {
                  colorize_log(f"{score:.3f}", "SUCCESS")})")
            print(f"{doc}")

    save_file({
        "query": query,
        "count": len(results),
        "results": results
    }, f"{OUTPUT_DIR}/results.json")

# Plateful

> Your AI kitchen agent that plans meals, orders groceries from Kroger, tracks your pantry, and guides you step-by-step with a voice companion to make cooking effortless.

<img src="./logo.png" alt="Plateful App" width="600">

---

## Authors

* [Zaid Madanat][linkedin-url-1]
* [Priyadarsi Mishra][linkedin-url-2]
* [Rajeev Duggirala][linkedin-url-3]
* [Ameen Mobedi][linkedin-url-4]

---

## Inspiration

The daily question of "What's for dinner?" often leads to decision fatigue, wasted food, and last-minute unhealthy choices. We were inspired to streamline the entire meal preparation process, from initial idea to finished plate. We envisioned a single, intelligent agent that could handle the mental load of planning, shopping, and tracking, allowing people to rediscover the joy of cooking without the usual stress. Plateful was born from the desire to make home cooking not just easier, but truly effortless and enjoyable. 🧑‍🍳

---

## What it does

Plateful is a comprehensive kitchen assistant, powered by a sophisticated two-agent system and delivered through a native mobile experience.

1.  **Personalized AI Meal Planning (Recipe Agent):** The **Recipe Agent**, built on the **Claude API**, is the core planner. It analyzes your user profile—including macros, daily calorie targets, and dietary restrictions—to generate a full week of personalized recipes. Using a **Retrieval-Augmented Generation (RAG)** architecture, it learns from your meal history stored in a **ChromaDB** vector store to make its suggestions smarter and more tailored over time.

2.  **Smart Grocery Ordering (Grocery Agent):** Once you approve a meal plan, the **Grocery Agent** takes over. It communicates directly with the Recipe Agent, receives the ingredient list, and uses tools to interact with the **Kroger Public API**. It populates your cart with the exact items and quantities needed, making your entire weekly shop ready for checkout with a single click. 🛒

3.  **Context-Aware Voice Companion:** Elevating the cooking experience, our voice companion offers hands-free guidance. Built with **LiveKit** for real-time audio streaming and **LlamaIndex** for context-awareness, it knows exactly which step you're on. You can ask questions and control the flow of the recipe with your voice, making the process interactive and seamless.

---

## How we built it

Plateful is built with a modern, mobile-first tech stack designed for a responsive user experience and a powerful, AI-driven backend.

* **Mobile App (Frontend):** A fully native iOS application built with **Swift** & **SwiftUI** to provide a fluid and intuitive user interface.

* **Backend:** A high-performance asynchronous backend built with **Python** & **FastAPI**, powering our agent-based system and managing all business logic.

* **AI & Language Models:**
    * **Core Agent Logic:** The **Claude API** serves as the foundational intelligence for both the Recipe and Grocery agents, handling reasoning, planning, and tool usage.
    * **Memory & Personalization:** We use a **Retrieval-Augmented Generation (RAG)** architecture with **ChromaDB** as the vector store to provide our agents with long-term memory of user preferences.
    * **Voice Context:** **LlamaIndex** is used to build the context-aware query engine for the voice companion, allowing it to understand the recipe state.

* **Database:**
    * **User Data:** A **SQL Database** (e.g., PostgreSQL) stores user profiles, dietary information, and application data.
    * **Vector Storage:** **ChromaDB** is used for embedding and storing user meal interaction history.

* **APIs & Real-Time Communication:**
    * **Groceries:** **Kroger Public API** for product lookup and cart management.
    * **Voice Streaming:** **LiveKit** for real-time, low-latency audio and data streaming for the voice companion.

---

## Challenges we ran into

1.  **Agent Reliability:** Ensuring the Claude-powered agents could reliably interpret user needs, generate valid recipes, and correctly format API calls to Kroger required extensive prompt engineering and structured output validation.

2.  **Real-Time Voice Context:** Managing the conversational state between the user, LiveKit, and the LlamaIndex query engine in real-time was complex. Achieving low latency and accurate context-awareness to avoid frustrating the user during a hands-on cooking task was a major hurdle.

3.  **RAG Pipeline Tuning:** Fine-tuning the process of embedding user feedback (recipe selections/skips) and ensuring the retrieval from ChromaDB consistently provided useful, non-repetitive context for the LLM was a significant challenge.

---

## Accomplishments that we're proud of

* Successfully creating a seamless, end-to-end workflow from AI-driven meal plan generation to a verified Kroger grocery cart with a single click.
* Building a sophisticated RAG pipeline that truly learns from user behavior, making meal planning increasingly personalized and effective with each use.
* Implementing a genuinely context-aware voice companion that provides a hands-free cooking experience that feels natural and truly helpful, not robotic.

---

## What we learned

* **The Power of Agentic Architecture:** Structuring our backend with distinct agents (Recipe, Grocery) built on a powerful LLM like Claude allows for modular, scalable, and highly capable systems.
* **State Management is Crucial:** For a feature like a voice companion, managing the state of the conversation and the recipe is just as important as the AI model itself. A small error can ruin the user experience.
* **Data is Key for Personalization:** A simple AI model can give generic answers, but integrating a RAG system with a vector database is what unlocks true, meaningful personalization that users notice and appreciate.

---

## What's next for Plateful

1.  **Multi-Store Support:** Expanding our grocery integration to include other major retailers like Safeway, Walmart, and Instacart.
2.  **Recipe Discovery & Import:** Allowing users to import recipes from their favorite websites and have Plateful automatically parse ingredients and instructions.
3.  **Android Application:** Developing a native Android version of Plateful to reach a wider audience.
4.  **Smart Appliance Integration:** Connecting with smart ovens, refrigerators, and other kitchen gadgets to further automate the cooking process.

---

## Disclaimer

The content, meal plans, and nutritional information generated by Plateful are for informational purposes only. You should not construe any such information as medical, dietary, or health advice. Always consult with a professional for any health or allergy-related concerns before making decisions based on information from this site.

[linkedin-url-1]: https://www.linkedin.com/in/zaid-madanat-8257001b3/
[linkedin-url-2]: https://www.linkedin.com/in/priyadarsi-mishra/
[linkedin-url-3]: https://www.linkedin.com/in/rajeev-duggirala-7307992b6/
[linkedin-url-4]: https://www.linkedin.com/in/aminmobedi/

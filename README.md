# OmniShopAgent
```
mermaid
graph TD
    %% --- 1. Core Components ---
    subgraph UserInput
        User["User (Identified by <b>Session ID</b>)"] 
    end
    
    subgraph ApplicationLayer
        API["<b>FastAPI Backend</b><br/>(Receives query + <b>Session ID</b>)"] 
    end
    
    subgraph OrchestrationLayer
        Agent["<b>Agent (LLM Brain / Router)</b><br/>Tool: LangChain<br/><i>'Which flow should I execute?'</i>"]
    end
    
    subgraph Tools
        T1["<b>Tool 1: VLM Reasoning</b><br/>(LLaVA / GPT-4V)"]
        T2["<b>Tool 2: Visual Search</b><br/>(CLIP Model)"]
        T3["<b>Tool 3: Text Search (RAG)</b><br/>(Text Embedding Model)"]
    end

    subgraph DataStores[Data Stores]
        Milvus[("<b>Vector DB (Milvus)</b><br/>- Image Embeddings<br/>- Text Embeddings")]
        Mongo[("<b>Metadata DB (MongoDB)</b><br/>- Price, Color, Name")]
        History[("<b>Chat History DB (Redis)</b><br/>- Stores history <b>per Session ID</b>")] 
    end

    %% --- 2. Start Flow ---
    User --> API
    API -- "Current Query + Session ID" --> Agent
    History -- "Past Conversation<br/>(using Session ID as Key)" --> Agent 
    
    %% --- 3. Agent routes to different workflows ---

    %% --- Flow 1: Text RAG ---
    subgraph Flow1 ["Flow 1: Text RAG (e.g., 'a 100% cotton blue shirt')"]
        direction LR
        Agent -- "<b>Flow 1</b>" --> F1_Start(T3: Text Search)
        F1_Start -- "1. Query Vector" --> Milvus
        Milvus -- "2. Return Top K 'noisy' IDs<br/>(incl. 'Red Cotton Shirt')" --> F1_GetMongo
        F1_GetMongo["Get Metadata"] --> Mongo
        Mongo -- "3. Return K 'noisy' Data Objects" --> F1_Augment
        F1_Augment["<b>Context Augmentation</b><br/>(Pass K results to LLM)"] --> F1_Generate
        F1_Generate["<b>Generation (Reasoning)</b><br/>LLM <b>reasons & filters</b> 'Red Shirt',<br/>then generates the answer"] --> API
    end

    %% --- Flow 2: Pure Visual Search ---
    subgraph Flow2 ["Flow 2: Pure Visual Search (e.g., [image] + 'find similar')"]
        direction LR
        Agent -- "<b>Flow 2</b>" --> F2_Start(T2: Visual Search)
        F2_Start -- "1. Image Vector" --> Milvus
        Milvus -- "2. Return Top K 'trusted' IDs" --> F2_GetMongo
        F2_GetMongo["Get Metadata"] --> Mongo
        Mongo -- "3. Return K 'trusted' Data Objects" --> F2_Augment
        F2_Augment["<b>Context Augmentation</b><br/>(Pass K results to LLM)"] --> F2_Generate
        F2_Generate["<b>Generation (Presenting)</b><br/>LLM <b>presents</b> trusted results<br/>(no filtering needed)"] --> API
    end
    
    %% --- Flow 3: Visual Search + Metadata Filter ---
    subgraph Flow3 ["Flow 3: Visual Search + Filter (e.g., [image] + '...but in blue')"]
        direction LR
        Agent -- "<b>Flow 3</b>" --> F3_Start(T2: Visual Search)
        F3_Start -- "1. Image Vector" --> Milvus
        Milvus -- "2. Return Top 100 IDs" --> F3_FilterMongo
        F3_FilterMongo["<b>Database Filter (MongoDB)</b><br/>(Query: WHERE ID IN [...]<br/>AND color == 'blue')"] --> Mongo
        Mongo -- "3. Return 3 'filtered' Data Objects" --> F3_Augment
        F3_Augment["<b>Context Augmentation</b><br/>(Pass 3 filtered results to LLM)"] --> F3_Generate
        F3_Generate["<b>Generation (Presenting)</b><br/>LLM formats the 3 results<br/>and generates the answer"] --> API
    end
    
    %% --- Flow 4: ReAct Loop (VLM -> RAG Chain) ---
    subgraph Flow4 ["Flow 4: ReAct Loop (e.g., [image] + 'find me floor lamps in this style')"]
        direction LR
        Agent -- "<b>Flow 4</b>" --> F4_Reason1["<b>Turn 1: Reason</b><br/>'I need the style first.<br/>I must call Tool 1 (VLM).'"]
        F4_Reason1 --> F4_Act1["<b>Turn 1: Act</b><br/>Call T1 (VLM)"]
        F4_Act1 --> T1
        T1 -- "<b>Observation:</b><br/>'This is a modern style lamp'" --> F4_Reason2
        F4_Reason2["<b>Turn 2: Reason</b><br/>'OK, style is modern.<br/>I must now call Tool 3 (RAG)<br/>to find modern floor lamps.'"]
        F4_Reason2 --> F4_Act2["<b>Turn 2: Act</b><br/>Call T3 (Text RAG)<br/>with new query"]
        F4_Act2 --> F4_RAGFlow("RAG Sub-Flow<br/>(Runs full Flow 1)")
        F4_RAGFlow -- "Final Answer" --> API
    end

    %% --- Flow 5: Conversational Memory ---
    subgraph Flow5 ["Flow 5: Conversational Memory (e.g., 'find matching chairs')"]
        direction LR
        Agent -- "<b>Flow 5</b>" --> F5_Reason["<b>Reason (with Memory)</b><br/>'Query is find matching chairs.<br/><b>Get history for Session ID.</b><br/>History says we just<br/>discussed a 'Mid-Century' sofa.<br/>I must call Tool 3 (RAG).'"] 
        F5_Reason --> F5_Act["<b>Act</b><br/>Call T3 (Text RAG)<br/>with query:<br/>'Mid-Century modern chairs'"]
        F5_Act --> F5_RAGFlow("RAG Sub-Flow<br/>(Runs full Flow 1)")
        F5_RAGFlow -- "Final Answer" --> API
    end
    
    %% --- 4. Final Response ---
    API -- "Final Answer" --> FinalResponse["User (Final Response)"]
    API -- "Save to DB<br/>(using Session ID as Key)" --> History 

    %% --- Style Definitions ---
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    class Agent fill:#e6f7ff,stroke:#0056b3,stroke-width:2px;
    class T1,T2,T3 fill:#fffbe6,stroke:#b8860b,stroke-width:2px;
    class Milvus,Mongo,History fill:#e6ffe6,stroke:#006400,stroke-width:2px;
    class API,FinalResponse fill:#f0e6ff,stroke:#663399,stroke-width:2px;
    class F1_Generate,F4_RAGFlow,F5_RAGFlow fill:#ffebeb,stroke:b30000;
    class F2_Generate,F3_Generate fill:#ebfbee,stroke:006400;
    class F4_Reason1,F4_Act1,F4_Reason2,F4_Act2,F5_Reason,F5_Act fill:f5f5f5,stroke:444;
```
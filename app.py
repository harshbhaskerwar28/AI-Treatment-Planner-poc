import streamlit as st
import os
import openai
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv()

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .card {
        background: #1a1a1a;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .loading-card {
        background: linear-gradient(45deg, #2a2a2a, #1a1a1a);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        border: 2px dashed #667eea;
    }
    
    .document-card {
        background: #2a2a2a;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
        font-size: 0.9rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .success-badge {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

class TreatmentPlannerRAG:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.documents = []
        
    def load_documents(self, data_folder="data"):
        """Load PDF files directly for OpenAI"""
        if not os.path.exists(data_folder):
            return False
        
        pdf_files = list(Path(data_folder).glob("*.pdf"))
        if not pdf_files:
            return False
        
        self.documents = []
        
        for pdf_file in pdf_files:
            # Store PDF file path for direct use with OpenAI
            self.documents.append({
                'file_path': str(pdf_file),
                'file_name': pdf_file.name
            })
        
        return True
    
    def upload_files_to_openai(self):
        """Upload PDF files to OpenAI for assistant"""
        uploaded_files = []
        
        for doc in self.documents:
            try:
                with open(doc['file_path'], 'rb') as file:
                    uploaded_file = self.client.files.create(
                        file=file,
                        purpose='assistants'
                    )
                    uploaded_files.append(uploaded_file.id)
            except Exception as e:
                st.error(f"Failed to upload {doc['file_name']}: {str(e)}")
        
        return uploaded_files

    def generate_treatment_plan(self, condition, patient_info=""):
        """Generate treatment plan using OpenAI Assistant with PDF files"""
        
        # Upload PDFs to OpenAI if not done already
        if not hasattr(self, 'uploaded_files'):
            self.uploaded_files = self.upload_files_to_openai()
        
        # Create assistant with file access
        assistant = self.client.beta.assistants.create(
            name="Medical Treatment Planner",
            instructions="""You are a senior medical consultant AI providing evidence-based treatment recommendations to healthcare professionals. 

Analyze the provided medical PDF files and create a comprehensive, professional treatment plan.

INSTRUCTIONS FOR TREATMENT PLAN:
Create a detailed, professional treatment plan following this exact structure. Base ALL recommendations on the provided medical literature.

## COMPREHENSIVE TREATMENT PLAN

### 1. CLINICAL ASSESSMENT & DIFFERENTIAL DIAGNOSIS
- Confirm primary diagnosis based on medical literature
- Rule out differential diagnoses mentioned in the sources
- Risk stratification according to clinical guidelines

### 2. EVIDENCE-BASED TREATMENT APPROACH
#### A. FIRST-LINE THERAPY
- Primary treatment recommendations with specific protocols
- Medication regimens with exact dosages, frequencies, and durations
- Reference the medical sources supporting these choices

#### B. ADJUNCTIVE TREATMENTS
- Supporting therapies based on literature evidence
- Non-pharmacological interventions with protocols

### 3. PHARMACOLOGICAL MANAGEMENT
#### PRESCRIBED MEDICATIONS:
For each medication, include:
- Generic and brand names
- Exact dosing regimen (dose, frequency, timing)
- Route of administration
- Duration of treatment
- Contraindications and precautions

#### MONITORING REQUIREMENTS:
- Laboratory parameters to track
- Monitoring schedule
- Target therapeutic ranges

### 4. PATIENT CARE PROTOCOLS
- Immediate management (24-48 hours)
- Short-term management (1-4 weeks)
- Long-term management (1-6 months+)

### 5. MULTIDISCIPLINARY CARE COORDINATION
- Specialist referral criteria
- Allied health professional involvement

### 6. PATIENT EDUCATION & COUNSELING POINTS
- Disease explanation
- Treatment rationale
- Medication compliance strategies
- Warning signs requiring immediate medical attention

### 7. FOLLOW-UP & MONITORING SCHEDULE
- Specific appointment timeline
- Parameters to assess at each visit

### 8. PROGNOSIS & EXPECTED OUTCOMES
- Short-term and long-term prognosis
- Quality of life considerations

### 9. SAFETY CONSIDERATIONS & RED FLAGS
- Critical warning signs requiring emergency care
- Medication-related adverse events to monitor

### 10. EVIDENCE SUMMARY
- Key clinical guidelines referenced
- Strength of evidence for main recommendations

CRITICAL REQUIREMENTS:
✓ Base ALL recommendations on the provided PDF medical literature
✓ Cite specific sources when making recommendations give the exact filename for the siteation like "Source: filename.pdf"
✓ Use professional medical terminology
✓ Include specific dosages, frequencies, and monitoring parameters
✓ Prioritize patient safety in all recommendations""",
            model="gpt-4o-mini",
            tools=[{"type": "file_search"}],
            tool_resources={
                "file_search": {
                    "vector_stores": [{
                        "file_ids": self.uploaded_files
                    }]
                }
            }
        )
        
        # Create thread and run
        thread = self.client.beta.threads.create()
        
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"""Based on the uploaded medical PDF files, create a comprehensive treatment plan for:

PATIENT CONDITION: {condition}
PATIENT INFORMATION: {patient_info if patient_info else "Not provided"}

Please analyze all the medical literature in the PDF files and provide evidence-based treatment recommendations."""
        )
        
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Wait for completion
        while run.status != "completed":
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run.status == "failed":
                return "Error: Treatment plan generation failed"
        
        # Get response
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        response = messages.data[0].content[0].text.value
        
        # Cleanup
        self.client.beta.assistants.delete(assistant.id)
        
        return response

def main():
    st.set_page_config(
        page_title="AI Treatment Planner",
        page_icon="🏥",
        layout="wide"
    )
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI Treatment Planner</h1>
        <p>AI-powered treatment planning using medical literature</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        st.markdown("""
        <div class="card">
            <h3>⚠️ Configuration Required</h3>
            <p>Please add your OpenAI API key to the .env file</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Initialize system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = TreatmentPlannerRAG()
        st.session_state.documents_loaded = False
    
    # Sidebar - Documents
    with st.sidebar:
        st.markdown("### 📚 Medical Documents")
        
        # Loading state for documents
        if not st.session_state.documents_loaded:
            st.markdown("""
            <div class="loading-card">
                <h4>🔄 Loading Documents...</h4>
                <p>Scanning /data folder for PDF files</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Load documents
            if st.session_state.rag_system.load_documents():
                st.session_state.documents_loaded = True
                st.rerun()
            else:
                st.markdown("""
                <div class="card">
                    <h4>📁 No Documents Found</h4>
                    <p>Create a 'data' folder and add PDF files</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show loaded documents
            doc_count = len(st.session_state.rag_system.documents)
            st.markdown(f"""
            <div class="success-badge">
                ✅ {doc_count} PDF files loaded
            </div>
            """, unsafe_allow_html=True)
            
            # Document cards
            for doc in st.session_state.rag_system.documents:
                st.markdown(f"""
                <div class="document-card">
                    📄 {doc['file_name']}
                </div>
                """, unsafe_allow_html=True)
            
            # Reload button
            if st.button("🔄 Reload Documents", use_container_width=True):
                st.session_state.documents_loaded = False
                st.rerun()
    
    # Main content area - Input fields
    col1, col2 = st.columns([1, 1])
    
    with col1:
        condition = st.text_area("Medical Condition", height=120, 
                                placeholder="e.g., Type 2 Diabetes, Hypertension...")
    
    with col2:
        patient_info = st.text_area("Patient Details", height=120, 
                                   placeholder="Age, gender, allergies, current medications...")
    
    # Generate button below inputs
    if st.button("🚀 Generate Treatment Plan", type="primary", use_container_width=True):
        if not condition:
            st.error("Please enter the medical condition")
        elif not st.session_state.rag_system.documents:
            st.error("Please load medical documents first")
        else:
            # Show loading animation
            with st.spinner("🤖 AI is analyzing medical literature and generating treatment plan..."):
                plan = st.session_state.rag_system.generate_treatment_plan(condition, patient_info)
                st.session_state.treatment_plan = plan
                st.success("✅ Treatment plan generated successfully!")
    
    # Display results
    if 'treatment_plan' in st.session_state:
        st.markdown("---")
        
        # Plan content
        st.markdown(st.session_state.treatment_plan)
        
        # Download button
        st.download_button(
            "💾 Download Treatment Plan",
            st.session_state.treatment_plan,
            file_name=f"treatment_plan_{condition.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )

if __name__ == "__main__":
    main()

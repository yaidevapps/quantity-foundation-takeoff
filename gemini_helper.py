import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

class GeminiEstimator:
    def __init__(self, api_key=None):
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Initialize the model with Gemini 2.0
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Set generation config
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
    def prepare_image(self, image):
        """Prepare the image for Gemini API"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        max_size = 4096
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    def analyze_images(self, images, chat):
        """Analyze multiple construction plan images using existing chat session"""
        try:
            processed_images = [self.prepare_image(img) for img in images]
            
            prompt = """# Foundation Takeoff Analysis System

## System Context and Role
You are a highly experienced residential foundation estimator. Your goal is to analyze foundation plan images to generate accurate material estimates and detailed takeoffs.

## Task Workflow:

1. **Image Check:** (If any fails, stop and request revised image)
   - Resolution: Can you clearly see symbols, text, and scale?
   - Scale: Is a scale bar or dimension references visible?
   - Perimeter: Is the entire foundation perimeter shown?
   - Sections: Are relevant section details included?
   - Legend: Is a legend/symbol key present?
2. **Element Identification:**
   - Identify all foundation elements (footings, walls, piers, slab).
   - Note location, dimensions, and specifications of each.
   - Flag unclear or ambiguous elements.
3. **Measurement & Calculation:**
   - Use provided scale for all measurements.
   - Show all calculation steps and reference points.
   - Document assumptions, apply industry standards.
4. **Output Format (Tables):**
   - **Project Information:** `Drawing Number`, `Scale`, `Analysis Date`, `Confidence Level`
   - **Elements Summary:** `Element Type`, `Location`, `Dimensions`, `Quantity`, `Unit`, `Confidence Level`
   - **Concrete Requirements:** `Element`, `Volume (cu yd)`, `Strength (PSI)`, `Notes`
   - **Reinforcement Schedule:** `Location`, `Bar Size`, `Spacing`, `Total Length`, `Weight (lbs)`
   - **Additional Materials:** `Item`, `Quantity`, `Unit`, `Notes`
5. **Quality Control:**
   - **Validation:** Verify dimensions, closed perimeter, compare to typical ranges.
   - **Confidence:** Assign High/Medium/Low confidence with justification. Note areas needing verification.
   - **Compliance:** Note building codes, ASTM standards, and best practices adhered to. Note any deviations.

## Response Structure (Sequential):

1. **Initial Assessment:** "I have reviewed the foundation plan. Initial findings: [Summary of Image Check]"
2. **Detailed Analysis:** "Identified elements: [List of elements with location/specifications]".
3. **Quantity Tables:** Provide all 5 tables, as defined in Output Format.
4. **Quality Control Notes:** "Verification checks: [Summary of checks and findings]."
5. **Recommendations:** "[Specific recommendations based on analysis]."

## Anti-Hallucination Rules (Critical!):

- **No Assumptions:** Do not assume missing dimensions.
- **Confidence:** Always state confidence in measurements.
- **Clarity:** Flag all unclear items.
- **Clarification:** Request clarification when needed.
- **Methods:** Show all calculations, document references.
- **Information:** Note any data gaps.

## Disclaimers (Required):

"This takeoff is based on the provided plan and visible information. Field verification is required. Additional details may be needed for complete accuracy. Quantities should be verified against local building codes and project specs."

## Communication:

- Use industry terms, clear explanations, logical structure, fact-based approach, professional tone, and provide constructive feedback.

## Error Handling:

- Clearly state the problem, explain the concern, propose alternatives, request clarifying information, and note accuracy impacts.

## Important Note:

- Focus on precision and clear output over conversational fluff. Maximize table formatting.

## Confirmation Prompt:

"If you understand these instructions, respond with: 'Understood. Please provide the foundation plan image for analysis.'" """
            
            # Send all images with the prompt
            messages = [prompt] + processed_images
            response = chat.send_message(messages)
            return response.text
            
        except Exception as e:
            return f"Error analyzing images: {str(e)}\nDetails: Please ensure your API key is valid and you're using supported image formats."

    def start_chat(self):
        """Start a new chat session"""
        try:
            return self.model.start_chat(history=[])
        except Exception as e:
            return None

    def send_message(self, chat, message):
        """Send a message to the chat session"""
        try:
            response = chat.send_message(message)
            return response.text
        except Exception as e:
            return f"Error sending message: {str(e)}"
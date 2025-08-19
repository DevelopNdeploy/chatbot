# Enhanced flow_bot.py with better administrative document handling

import json
import csv
import os
import re
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from typing import Tuple, Dict
from sentence_transformers import SentenceTransformer, util
import numpy as np


class FlowBasedChatbot:
    def __init__(self):
        self.intents = self.load_intents()
        self.course_data = self.load_course_data()

        # Load embedding model (lightweight but powerful)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Precompute embeddings for intent patterns
        self.intent_embeddings = self.precompute_intent_embeddings()

        self.embedding_cache = {}

        # Administrative document keywords (high priority)
        self.admin_keywords = {
            'migration_certificate': ['migration', 'transfer certificate', 'leaving certificate'],
            'degree_certificate': ['degree certificate', 'original degree', 'convocation certificate'],
            'transcript_request': ['transcript', 'transcripts', 'official transcript', 'mark sheets'],
            'verification_services': ['verification', 'certificate verification', 'document verification']
        }

        self.conversation_flow = {
            'current_intent': None,
            'previous_intent': None,
            'course_interest': None,
            'student_profile': {
                'name': None,
                'education_level': None,
                'preferred_courses': [],
                'concerns_addressed': [],
                'engagement_level': 0
            },
            'conversation_history': [],
            'intent_history': [],
            'session_start': datetime.now()
        }

    def load_intents(self) -> Dict:
        """Load intents with related_intents support"""
        try:
            with open(os.path.join(os.path.dirname(__file__), 'intentsgr1.json'), 'r', encoding='utf-8') as file:
                data = json.load(file)

            for intent in data["intents"]:
                if "related_intents" not in intent:
                    intent["related_intents"] = []

            return data
        except FileNotFoundError:
            print("Warning: intents.json not found")
            return {"intents": []}

    def load_course_data(self) -> Dict:
        """Load additional course data if available"""
        course_data = {}
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            csv_path = os.path.join(base_dir, 'data', 'courses.csv')
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    keyword = row['keyword'].lower()
                    course_data[keyword] = {
                        'response': row.get('response', ''),
                        'related_keywords': row.get('related_keywords', '').split(',')
                    }
        except FileNotFoundError:
            print("Note: No additional course data file found")
        return course_data

    def check_administrative_intent(self, user_input: str) -> Optional[str]:
        """Check for administrative document requests first (high priority)"""
        user_lower = user_input.lower()
        
        # Direct matches for admin documents
        for intent_tag, keywords in self.admin_keywords.items():
            for keyword in keywords:
                if keyword in user_lower:
                    return intent_tag
        
        # Special handling for combined keywords
        if 'migration' in user_lower and 'certificate' in user_lower:
            return 'migration_certificate'
        
        if 'degree' in user_lower and ('certificate' in user_lower or 'original' in user_lower):
            return 'degree_certificate'
            
        if ('transcript' in user_lower or 'mark sheet' in user_lower) and ('official' in user_lower or 'certified' in user_lower):
            return 'transcript_request'
            
        return None

    # def analyze_user_intent(self, user_input: str) -> Tuple[str, Dict, float]:
    #     """Enhanced intent analysis with administrative document priority"""
    #     user_input_lower = user_input.lower()
        
    #     # STEP 1: Check for administrative documents first (highest priority)
    #     admin_intent = self.check_administrative_intent(user_input)
    #     if admin_intent:
    #         intent_data = self.find_intent_by_tag(admin_intent)
    #         if intent_data:
    #             return admin_intent, intent_data, 0.95
        
    #     # STEP 2: Regular intent matching
    #     best_match = None
    #     best_score = 0
        
    #     user_tokens = self.tokenize_and_clean(user_input_lower)
        
    #     for intent in self.intents["intents"]:
    #         score = self.calculate_intent_score(user_tokens, user_input_lower, intent)
            
    #         # Context bonus for related intents
    #         if (self.conversation_flow['current_intent'] and 
    #             self.conversation_flow['current_intent'] in intent.get("related_intents", [])):
    #             score += 0.1
                
    #         if score > best_score:
    #             best_score = score
    #             best_match = intent
        
    #     # STEP 3: Fallback to contextual matching
    #     if not best_match or best_score < 0.3:
    #         contextual_intent = self.get_contextual_intent(user_input)
    #         if contextual_intent:
    #             return contextual_intent["tag"], contextual_intent, 0.7
        
    #     # STEP 4: Final fallback to greeting or noanswer
    #     if not best_match or best_score < 0.2:
    #         # Check if it's a greeting
    #         greeting_words = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'namaste']
    #         if any(word in user_input_lower for word in greeting_words):
    #             greeting_intent = self.find_intent_by_tag("greeting")
    #             if greeting_intent:
    #                 return "greeting", greeting_intent, 0.8
            
    #         # Otherwise, use noanswer
    #         noanswer_intent = self.find_intent_by_tag("noanswer")
    #         if noanswer_intent:
    #             return "noanswer", noanswer_intent, 0.1
        
    #     return best_match["tag"], best_match, best_score


    def get_user_embedding(self, text):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        vec = self.embedding_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        self.embedding_cache[text] = vec
        return vec



    def precompute_intent_embeddings(self):
        """Precompute embeddings for all patterns in intents"""
        intent_embeddings = {}
        for intent in self.intents["intents"]:
            patterns = intent.get("patterns", [])
            if patterns:
                vectors = self.embedding_model.encode(patterns, convert_to_tensor=True, show_progress_bar=False)
                intent_embeddings[intent["tag"]] = vectors
        return intent_embeddings
    


    def analyze_user_intent(self, user_input: str) -> Tuple[str, Dict, float]:
        """Enhanced intent analysis with admin priority + embeddings"""
        user_input_lower = user_input.lower()
        
        # STEP 1: Admin priority check
        admin_intent = self.check_administrative_intent(user_input)
        if admin_intent:
            intent_data = self.find_intent_by_tag(admin_intent)
            if intent_data:
                return admin_intent, intent_data, 0.95

        # STEP 2: Rule-based matching (your existing logic)
        best_match = None
        best_score = 0
        user_tokens = self.tokenize_and_clean(user_input_lower)
        
        for intent in self.intents["intents"]:
            score = self.calculate_intent_score(user_tokens, user_input_lower, intent)

            # Context bonus
            if (self.conversation_flow['current_intent'] and
                self.conversation_flow['current_intent'] in intent.get("related_intents", [])):
                score += 0.1

            if score > best_score:
                best_score = score
                best_match = intent

        # STEP 3: Embedding similarity fallback (only if weak score)
        if not best_match or best_score < 0.5:
            user_vec = self.self.get_user_embedding(user_input)
            best_sim = 0
            best_intent = None

            for tag, vectors in self.intent_embeddings.items():
                sim = util.cos_sim(user_vec, vectors).max().item()
                if sim > best_sim:
                    best_sim = sim
                    best_intent = tag
            
            if best_intent and best_sim > 0.55:  # Threshold for semantic similarity
                intent_data = self.find_intent_by_tag(best_intent)
                return best_intent, intent_data, best_sim

        # STEP 4: Contextual fallback
        if not best_match or best_score < 0.3:
            contextual_intent = self.get_contextual_intent(user_input)
            if contextual_intent:
                return contextual_intent["tag"], contextual_intent, 0.7

        # STEP 5: Greeting or noanswer
        if not best_match or best_score < 0.2:
            greeting_words = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'namaste']
            if any(word in user_input_lower for word in greeting_words):
                greeting_intent = self.find_intent_by_tag("greeting")
                if greeting_intent:
                    return "greeting", greeting_intent, 0.8

            noanswer_intent = self.find_intent_by_tag("noanswer")
            if noanswer_intent:
                return "noanswer", noanswer_intent, 0.1

        return best_match["tag"], best_match, best_score


    def find_intent_by_tag(self, tag: str) -> Optional[Dict]:
        """Helper to find intent by tag"""
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return intent
        return None

    def tokenize_and_clean(self, text: str) -> List[str]:
        """Extract meaningful tokens from text"""
        stop_words = {'tell', 'me', 'about', 'what', 'is', 'how', 'can', 'you', 'please', 'i', 'want', 'to', 'know', 'the', 'a', 'an'}
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if token not in stop_words and len(token) > 2]

    def calculate_intent_score(self, user_tokens: List[str], user_input: str, intent: Dict) -> float:
        """Calculate semantic relevance score for an intent"""
        max_score = 0
        
        # 1. Exact pattern matching (highest priority)
        for pattern in intent.get("patterns", []):
            if pattern.lower() in user_input:
                return 0.9  # Very high confidence for exact matches
        
        # 2. Keyword-based scoring
        # keyword_score = self.get_keyword_score(user_tokens, user_input, intent)
        # max_score = max(max_score, keyword_score)
        
        # 3. Pattern token overlap scoring
        pattern_score = self.get_pattern_score(user_tokens, user_input, intent)
        max_score = max(max_score, pattern_score)
        
        # 4. Tag-based scoring
        tag_score = self.get_tag_score(user_input, intent)
        max_score = max(max_score, tag_score)
        
        return max_score

    # def get_keyword_score(self, user_tokens: List[str], user_input: str, intent: Dict) -> float:
    #     """Score based on keyword matching"""
    #     keywords = intent.get("keywords", [])
    #     if not keywords:
    #         return 0
        
    #     matches = 0
    #     total_keywords = len(keywords)
        
    #     for keyword in keywords:
    #         if keyword in user_input:
    #             matches += 1
    #         elif any(keyword in token or token in keyword for token in user_tokens):
    #             matches += 0.7
        
    #     if matches > 0:
    #         base_score = 0.6 + (matches / total_keywords) * 0.3
    #         return min(base_score, 0.95)
        
    #     return 0

    def get_pattern_score(self, user_tokens: List[str], user_input: str, intent: Dict) -> float:
        """Score based on flexible pattern matching"""
        patterns = intent.get("patterns", [])
        if not patterns:
            return 0
        
        max_pattern_score = 0
        
        for pattern in patterns:
            pattern_tokens = self.tokenize_and_clean(pattern)
            if not pattern_tokens:
                continue
            
            # Token overlap scoring
            overlap = len(set(user_tokens) & set(pattern_tokens))
            if overlap > 0:
                score = 0.4 + (overlap / max(len(pattern_tokens), len(user_tokens))) * 0.4
                max_pattern_score = max(max_pattern_score, score)
        
        return max_pattern_score

    def get_tag_score(self, user_input: str, intent: Dict) -> float:
        """Score based on direct tag mention"""
        tag = intent["tag"]
        tag_words = tag.lower().replace('_', ' ').split()
        
        matches = sum(1 for word in tag_words if word in user_input)
        if matches > 0:
            return 0.4 + (matches / len(tag_words)) * 0.3
        
        return 0

    def get_contextual_intent(self, user_input: str) -> Optional[Dict]:
        """Enhanced contextual matching"""
        keyword_map = {
            # Administrative documents (high priority)
            'migration': 'migration_certificate',
            'degree': 'degree_certificate', 
            'transcript': 'transcript_request',
            'certificate': 'migration_certificate',  # Default to migration for generic "certificate"
            
            # Course related
            'engineering': 'btech_selection',
            'btech': 'btech_selection',
            'computer': 'bca_selection',
            'management': 'mba_selection',
            'business': 'mba_selection',
            'medical': 'mbbs_selection',
            'doctor': 'mbbs_selection',
            'mba': 'mba_selection',
            'bca': 'bca_selection',
            'mbbs': 'mbbs_selection',
            
            # Process related
            'fees': 'fees_structure',
            'cost': 'fees_structure',
            'eligibility': 'course_discovery',
            'admission': 'admission_process',
            'apply': 'admission_process',
            'scholarship': 'scholarships',
            'placement': 'placement_opportunities',
            'job': 'placement_opportunities',
            'hostel': 'hostel_facilities',
            'library': 'library_facilities',
            'sports': 'sports_facilities'
        }
        
        user_tokens = self.tokenize_and_clean(user_input)
        
        for token in user_tokens:
            if token in keyword_map:
                intent_tag = keyword_map[token]
                intent = self.find_intent_by_tag(intent_tag)
                if intent:
                    return intent
        
        return None

    def update_student_profile(self, intent_tag: str, user_input: str):
        """Update profile with course interest and concerns"""
        profile = self.conversation_flow['student_profile']

        # Course interests
        course_intents = ['btech_selection', 'mba_selection', 'bca_selection', 'mbbs_selection']
        if intent_tag in course_intents:
            course_name = intent_tag.replace('_selection', '').upper()
            if course_name not in profile['preferred_courses']:
                profile['preferred_courses'].append(course_name)
            self.conversation_flow['course_interest'] = course_name

        # Document requests
        doc_intents = ['migration_certificate', 'degree_certificate', 'transcript_request']
        if intent_tag in doc_intents:
            concern = intent_tag.replace('_', ' ').title()
            if concern not in profile['concerns_addressed']:
                profile['concerns_addressed'].append(concern)

        # Other concerns
        concern_intents = ['fees_structure', 'admission_process', 'hostel_facilities', 'placement_opportunities']
        if intent_tag in concern_intents:
            concern = intent_tag.replace('_', ' ').title()
            if concern not in profile['concerns_addressed']:
                profile['concerns_addressed'].append(concern)

        profile['engagement_level'] += 1

        # Extract name if mentioned
        if not profile['name']:
            match = re.search(r'\b(?:my name is|i am|i\'m)\s+([A-Za-z]+)', user_input, re.IGNORECASE)
            if match:
                profile['name'] = match.group(1).title()

    def update_conversation_flow(self, intent_tag: str, intent_data: Dict):
        """Update intent and history"""
        self.conversation_flow['previous_intent'] = self.conversation_flow['current_intent']
        self.conversation_flow['current_intent'] = intent_tag

        # Update intent history (last 3)
        history = self.conversation_flow['intent_history']
        history.append(intent_tag)
        if len(history) > 3:
            history.pop(0)

    def get_personalized_response(self, intent_data: Dict) -> str:
        """Personalize response with name and context"""
        response = random.choice(intent_data["responses"])
        profile = self.conversation_flow['student_profile']
        
        # Add name if available
        if profile['name']:
            response = response.replace("Hello!", f"Hello {profile['name']}!")
            response = response.replace("Hi there!", f"Hi {profile['name']}!")
        
        return response

    # def get_recommendations(self, intent_data: Dict) -> List[str]:
    #     """Get contextual recommendations"""
    #     intent_tag = intent_data["tag"]
        
    #     # Get related intents from JSON
    #     related_intents = intent_data.get("related_intents", [])
    #     recommendations = []
        
    #     # Convert related intents to user-friendly suggestions
    #     for related_tag in related_intents[:3]:
    #         if related_tag == "document_checklist":
    #             recommendations.append("What documents do I need?")
    #         elif related_tag == "verification_services":
    #             recommendations.append("How to verify certificates?")
    #         elif related_tag == "examination_office":
    #             recommendations.append("Contact examination office")
    #         elif "btech" in related_tag:
    #             recommendations.append("Tell me about BTech Program")
    #         elif "mba" in related_tag:
    #             recommendations.append("Tell me about MBA Program")
    #         elif "fees" in related_tag:
    #             recommendations.append("What are the fees?")
    #         elif "admission" in related_tag:
    #             recommendations.append("How to apply?")
        
    #     # Add some general suggestions if we don't have enough
    #     general_suggestions = [
    #         "Show me courses",
    #         "Tell me about admissions", 
    #         "What are the fees?",
    #         "Campus facilities"
    #     ]
        
    #     # Combine and deduplicate
    #     all_suggestions = recommendations + general_suggestions
    #     seen = set()
    #     final_suggestions = []
    #     for suggestion in all_suggestions:
    #         if suggestion not in seen and len(final_suggestions) < 4:
    #             seen.add(suggestion)
    #             final_suggestions.append(suggestion)
        
    #     return final_suggestions


    def get_recommendations(self, intent_data: Dict) -> List[str]:
        """Smarter contextual recommendations using embeddings + profile context"""
        intent_tag = intent_data["tag"]
        recommendations = []

        # STEP 1: Related intents from JSON
        related_intents = intent_data.get("related_intents", [])
        for related_tag in related_intents[:3]:
            intent = self.find_intent_by_tag(related_tag)
            if not intent:
                continue
            # Pick a representative response pattern for recommendation
            if intent.get("patterns"):
                rec = f"Ask about: {random.choice(intent['patterns'])}"
                recommendations.append(rec)
            else:
                rec = f"Learn about {related_tag.replace('_', ' ').title()}"
                recommendations.append(rec)

        # STEP 2: Profile-aware recommendations
        profile = self.conversation_flow['student_profile']
        course_interest = self.conversation_flow.get('course_interest')

        if course_interest and "course" not in intent_tag:
            recommendations.append(f"Details about {course_interest} program")

        if profile['preferred_courses']:
            for course in profile['preferred_courses'][:2]:
                recommendations.append(f"Eligibility criteria for {course}")
        if "certificate" in intent_tag or "transcript" in intent_tag:
            recommendations.append("How to verify my documents?")
            recommendations.append("What documents are needed for this process?")

        # STEP 3: Semantic suggestions (using embeddings)
        user_history = [h['user_input'] for h in self.conversation_flow['conversation_history'][-3:]]
        if user_history:
            user_vec = self.embedding_model.encode(user_history[-1], convert_to_tensor=True, show_progress_bar=False)
            best_matches = []
            for intent in self.intents["intents"]:
                if intent["tag"] == intent_tag:
                    continue
                if intent.get("patterns"):
                    sim = util.cos_sim(user_vec, self.embedding_model.encode(intent["patterns"], convert_to_tensor=True)).max().item()
                    best_matches.append((sim, intent))
            # Pick top 2 semantically close intents
            best_matches.sort(reverse=True, key=lambda x: x[0])
            for sim, intent in best_matches[:2]:
                if intent.get("patterns"):
                    recommendations.append(f"Also: {random.choice(intent['patterns'])}")

        # STEP 4: Add general fallback suggestions if too few
        general_suggestions = [
            "Show me courses",
            "Tell me about admissions",
            "What are the fees?",
            "Campus facilities"
        ]
        while len(recommendations) < 4 and general_suggestions:
            rec = general_suggestions.pop(0)
            if rec not in recommendations:
                recommendations.append(rec)

        # STEP 5: Deduplicate & limit
        seen = set()
        final_recommendations = []
        for rec in recommendations:
            if rec not in seen and len(final_recommendations) < 5:  # max 5 suggestions
                seen.add(rec)
                final_recommendations.append(rec)

        return final_recommendations

    def process_message(self, user_input: str) -> Dict:
        """Main processing method"""
        intent_tag, intent_data, confidence = self.analyze_user_intent(user_input)
        
        self.update_student_profile(intent_tag, user_input)
        self.update_conversation_flow(intent_tag, intent_data)
        
        response = self.get_personalized_response(intent_data)
        recommendations = self.get_recommendations(intent_data)
        
        # Store conversation history
        self.conversation_flow['conversation_history'].append({
            'user_input': user_input,
            'intent': intent_tag,
            'confidence': confidence,
            'response': response,
            'timestamp': datetime.now()
        })

        return {
            'response': response,
            'recommendations': recommendations,
            'intent': intent_tag,
            'confidence': confidence * 100,  # Convert to percentage
            'current_intent': intent_tag,
            'conversation_summary': self.get_conversation_summary()
        }

    def get_conversation_summary(self) -> Dict:
        """Return current conversation state"""
        return {
            'current_intent': self.conversation_flow['current_intent'],
            'course_interest': self.conversation_flow['course_interest'],
            'student_profile': self.conversation_flow['student_profile'],
            'conversation_length': len(self.conversation_flow['conversation_history']),
            'session_duration': (datetime.now() - self.conversation_flow['session_start']).seconds
        }

    def reset_conversation(self):
        """Reset for new session"""
        self.conversation_flow = {
            'current_intent': None,
            'previous_intent': None,
            'course_interest': None,
            'student_profile': {
                'name': None,
                'education_level': None,
                'preferred_courses': [],
                'concerns_addressed': [],
                'engagement_level': 0
            },
            'conversation_history': [],
            'intent_history': [],
            'session_start': datetime.now()
        }
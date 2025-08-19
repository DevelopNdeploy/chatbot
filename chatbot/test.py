admin_keywords = {
            'migration_certificate': ['migration', 'transfer certificate', 'leaving certificate'],
            'degree_certificate': ['degree certificate', 'original degree', 'convocation certificate'],
            'transcript_request': ['transcript', 'transcripts', 'official transcript', 'mark sheets'],
            'verification_services': ['verification', 'certificate verification', 'document verification']
        }

def check_administrative_intent( user_input: str):
        """Check for administrative document requests first (high priority)"""
        user_lower = user_input.lower()
        
        # Direct matches for admin documents
        for intent_tag, keywords in admin_keywords.items():
            for keyword in keywords:
                if user_lower in keyword:
                    print(intent_tag)
        
        # Special handling for combined keywords
        if 'migration' in user_lower and 'certificate' in user_lower:
            return 'migration_certificate'
        
        if 'degree' in user_lower and ('certificate' in user_lower or 'original' in user_lower):
            return 'degree_certificate'
            
        if ('transcript' in user_lower or 'mark sheet' in user_lower) and ('official' in user_lower or 'certified' in user_lower):
            return 'transcript_request'
            
        return None

print(check_administrative_intent('certificate'))
# print(admin_keywords.items())
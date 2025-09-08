# app/utils/metadata_filter.py
from typing import List, Dict
from datetime import datetime

class MetadataFilter:
    """Advanced filtering and weighting based on sermon metadata"""
    
    def __init__(self):
        self.topic_keywords = {
            'worship': ['worship', 'praise', 'adoration', 'singing', 'music'],
            'prayer': ['prayer', 'pray', 'intercession', 'petition'],
            'relationships': ['marriage', 'family', 'friendship', 'relationship'],
            'leadership': ['leadership', 'leader', 'authority', 'responsibility'],
            'evangelism': ['evangelism', 'witness', 'testimony', 'sharing faith'],
            'discipleship': ['discipleship', 'following', 'growth', 'maturity'],
            'service': ['service', 'ministry', 'helping', 'volunteering'],
            'stewardship': ['giving', 'tithe', 'stewardship', 'generosity'],
            'forgiveness': ['forgiveness', 'mercy', 'grace', 'pardon'],
            'hope': ['hope', 'future', 'eternal', 'heaven']
        }
    
    def detect_sermon_topics(self, content: str) -> List[str]:
        """Detect topics in sermon content"""
        content_lower = content.lower()
        detected_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
            if keyword_count >= 2:
                detected_topics.append(topic)
        
        return detected_topics
    
    def create_advanced_filters(self, query: str, user_preferences: Dict = None) -> Dict:
        """Create advanced filtering criteria"""
        filters = {
            'pastor_weights': {},
            'topic_weights': {},
            'exclude_topics': [],
            'boost_recent': False
        }
        
        query_topics = self.detect_sermon_topics(query)
        
        for topic in query_topics:
            filters['topic_weights'][topic] = 1.5
        
        if user_preferences:
            for pastor in user_preferences.get('favorite_pastors', []):
                filters['pastor_weights'][pastor] = 1.3
            
            for topic in user_preferences.get('preferred_topics', []):
                filters['topic_weights'][topic] = filters['topic_weights'].get(topic, 1.0) + 0.2
            
            filters['exclude_topics'] = user_preferences.get('avoid_topics', [])
            filters['boost_recent'] = user_preferences.get('prefer_recent', False)
        
        return filters
    
    def apply_metadata_weighting(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply metadata-based weighting to search results"""
        for result in results:
            metadata = result['metadata']
            content = result['content']
            
            base_score = result.get('relevance_score', result.get('combined_score', 0.5))
            weighted_score = base_score
            
            # Apply pastor weighting
            pastor = metadata.get('pastor', '')
            if pastor in filters['pastor_weights']:
                weighted_score *= filters['pastor_weights'][pastor]
            
            # Apply topic weighting
            sermon_topics = self.detect_sermon_topics(content)
            for topic in sermon_topics:
                if topic in filters['topic_weights']:
                    weighted_score *= filters['topic_weights'][topic]
                elif topic in filters['exclude_topics']:
                    weighted_score *= 0.3
            
            # Apply date weighting
            if filters['boost_recent']:
                try:
                    from app.sermon_processor import SermonProcessor
                    processor = SermonProcessor()
                    date_obj = processor.parse_date_for_sorting(metadata.get('date', ''))
                    days_old = (datetime.now() - date_obj).days
                    
                    if days_old < 180:
                        recency_boost = 1.0 + (180 - days_old) / 180 * 0.3
                        weighted_score *= recency_boost
                except:
                    pass
            
            result['weighted_score'] = min(weighted_score, 2.0)
            result['detected_topics'] = sermon_topics
        
        results.sort(key=lambda x: x['weighted_score'], reverse=True)
        return results
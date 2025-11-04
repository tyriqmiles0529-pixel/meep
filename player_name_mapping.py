"""
Player Name Mapping - Fix name mismatches between sportsbooks and NBA API

Common issues:
- Nicknames vs full names
- Periods in names (C.J. vs CJ)
- Accents/special characters
- Hyphenated names
"""

# Map: Sportsbook name -> NBA API name
PLAYER_NAME_MAPPING = {
    # Common mismatches found
    'C.J. McCollum': 'CJ McCollum',
    'Dennis Schroder': 'Dennis Schröder',
    'A.J. Green': 'AJ Green',
    'Nikola Jovic': 'Nikola Jović',
    'Moussa Diabate': 'Moussa Diabaté',
    
    # Players with Jr./III
    'Derrick Jones': 'Derrick Jones Jr.',
    'Herb Jones': 'Herbert Jones',
    'P.J. Washington': 'PJ Washington',
    
    # Nickname variations
    'Nikola Jokic': 'Nikola Jokić',
    'Nicolas Claxton': 'Nic Claxton',
    'Jusuf Nurkic': 'Jusuf Nurkić',
    'Kristaps Porzingis': 'Kristaps Porziņģis',
    
    # More initials
    'R.J. Barrett': 'RJ Barrett',
    'O.G. Anunoby': 'OG Anunoby',
    'T.J. McConnell': 'TJ McConnell',
    'J.J. Redick': 'JJ Redick',
    'K.J. Martin': 'KJ Martin',
    
    # International players with accents
    'Luka Doncic': 'Luka Dončić',
    'Nikola Vucevic': 'Nikola Vučević',
    'Bojan Bogdanovic': 'Bojan Bogdanović',
    'Bogdan Bogdanovic': 'Bogdan Bogdanović',
    'Goran Dragic': 'Goran Dragić',
    'Jonas Valanciunas': 'Jonas Valančiūnas',
}

def normalize_player_name(name):
    """
    Normalize player name for matching
    
    Returns: (normalized_name, original_name)
    """
    # Check mapping first
    if name in PLAYER_NAME_MAPPING:
        return PLAYER_NAME_MAPPING[name], name
    
    # Normalize: remove periods, lowercase
    normalized = name.replace('.', '').replace('-', ' ').strip()
    
    return normalized, name

def find_player_id(player_name, all_players):
    """
    Try multiple strategies to find player ID
    
    Args:
        player_name: Name from sportsbook
        all_players: List of NBA players from nba_api (format: [{'id': 123, 'full_name': 'Name'}])
    
    Returns:
        player_id or None
    """
    # Strategy 1: Check mapping first
    if player_name in PLAYER_NAME_MAPPING:
        mapped_name = PLAYER_NAME_MAPPING[player_name]
        for p in all_players:
            if p['full_name'].lower() == mapped_name.lower():
                return p['id']
    
    # Strategy 2: Exact match
    for p in all_players:
        if p['full_name'].lower() == player_name.lower():
            return p['id']
    
    # Strategy 3: Remove periods (C.J. -> CJ)
    no_periods = player_name.replace('.', '').replace('  ', ' ').strip()
    for p in all_players:
        if p['full_name'].lower() == no_periods.lower():
            return p['id']
    
    # Strategy 4: Try first + last name only
    parts = player_name.replace('.', '').split()
    if len(parts) >= 2:
        first_last = f"{parts[0]} {parts[-1]}"
        for p in all_players:
            p_parts = p['full_name'].split()
            if len(p_parts) >= 2:
                p_first_last = f"{p_parts[0]} {p_parts[-1]}"
                if p_first_last.lower() == first_last.lower():
                    return p['id']
    
    # Strategy 5: Partial match (last name only for unique cases)
    if len(parts) >= 2:
        last_name = parts[-1].lower()
        matches = [p for p in all_players if p['full_name'].lower().endswith(last_name)]
        if len(matches) == 1:  # Only if unique
            return matches[0]['id']
    
    return None


# Print mapping for reference
if __name__ == '__main__':
    print("=" * 70)
    print("PLAYER NAME MAPPINGS")
    print("=" * 70)
    print("\nDefined mappings:")
    for sportsbook_name, nba_name in sorted(PLAYER_NAME_MAPPING.items()):
        print(f"  '{sportsbook_name}' -> '{nba_name}'")
    
    print(f"\nTotal mappings: {len(PLAYER_NAME_MAPPING)}")

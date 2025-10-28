import requests
from bs4 import BeautifulSoup
import urllib.parse

def get_game_image_url(game_name):
    """
    Recherche l'image d'un jeu via une recherche web basique
    
    Args:
        game_name (str): Nom du jeu vidéo
    
    Returns:
        str: URL de l'image ou message d'erreur
    """
    try:
        # Encode le nom du jeu pour l'URL
        search_query = urllib.parse.quote(f"{game_name} video game cover")
        
        # URL de recherche
        url = f"https://www.bing.com/images/search?q={search_query}"
        
        # En-têtes pour simuler un navigateur
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Fait la requête
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            # Parse le HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Cherche la première image
            # Note: les sélecteurs peuvent nécessiter des ajustements selon le site
            image_elements = soup.select('img.mimg')
            
            if image_elements:
                for img in image_elements:
                    if 'src' in img.attrs:
                        return img['src']
            
            return "Aucune image trouvée"
        else:
            return f"Erreur: Status code {response.status_code}"
            
    except Exception as e:
        return f"Erreur: {str(e)}"


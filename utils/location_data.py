import geonamescache

gc = geonamescache.GeonamesCache()
countries = gc.get_countries()
cities = gc.get_cities()

def get_country_list():
    return [(code, data['name']) for code, data in countries.items()]

def get_cities_for_country(country_code):
    return [
        (city['name'], float(city['latitude']), float(city['longitude']))
        for city in cities.values()
        if city['countrycode'] == country_code
    ]

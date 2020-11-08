# Uklanjanje projektivne distorzije

Dodatni domaci zadatak

## Instalacija

Potrebni su sledeci python moduli:
- cv2
- PIL
- matplotlib
- numpy

## Pokretanje programa

Dovoljno je pokrenuti program sa komandom `python main.py`.

## Koriscenje programa

Program ce prvo traziti da kliknete na 4 tacke koje ce predstavljati cetvorougao koji treba da se preslika u pravougaonik. 
Bice vam prikazana slika na kojoj cete moci da vidite koje ste tacke odabrali

Kada odaberete 4 tacke, potrebno je da zatvorite sve prozore sa slikama i program ce nastaviti sa radom.

Program ce ponovo traziti od vas da kliknete ovaj put na 2 tacke koje ce predstavljati dijagonalu pravougaonika u koji treba da se slika nas cetvorougao.
Bice vam prikazana slika na kojoj cete moci da vidite koje ste tacke odabrali.

Kada odaberete 2 tacke, potrebno je da zatvorite sve prozore sa slikama i program ce nastaviti sa radom.

Program zatim racuna matricu preslikavanja i prikazuje originalnu sliku i sliku sa ispravljenom distorzijom.

Program u konzoli ispisujete sta treba da radite.

## Koriscenje bez kliktanja

U funkciji `remove_distortion_naive` je naznaceno na koji nacin treba da pozovete program ako zelite da radi sa unapred poznatim tackama.




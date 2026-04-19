import random
import string
import os
import pandas as pd

random.seed(42)

FEMALE_NAMES = [
    "Anna", "Emma", "Mia", "Hanna", "Lena", "Lea", "Laura", "Julia", "Sarah", "Marie",
    "Sophie", "Lisa", "Katharina", "Christina", "Sandra", "Nicole", "Sabine", "Monika",
    "Petra", "Stefanie", "Claudia", "Daniela", "Elisabeth", "Franziska", "Gabriele",
    "Heike", "Inge", "Jana", "Karin", "Martina", "Nadine", "Olivia", "Paula", "Renate",
    "Sonja", "Tanja", "Ulrike", "Vera", "Yvonne", "Birgit", "Brigitte", "Carola",
    "Charlotte", "Diana", "Elke", "Eva", "Gisela", "Helga", "Ilse", "Johanna",
    "Katrin", "Kirsten", "Kristina", "Lotte", "Luisa", "Maria", "Miriam", "Natalie",
    "Nina", "Nora", "Ricarda", "Rosa", "Silke", "Simone", "Susanne", "Theresa",
    "Ursula", "Waltraud", "Anja", "Antje", "Barbara", "Beate", "Bettina", "Cornelia",
    "Dorothea", "Edith", "Friederike", "Gerda", "Hannelore", "Hildegard", "Irene",
    "Jutta", "Margit", "Marianne", "Martha", "Meike", "Melanie", "Michaela",
    "Ruth", "Silvia", "Swantje", "Wiebke", "Zoe", "Alina", "Amelie", "Clara",
    "Elena", "Emilia", "Ida", "Jasmin", "Josephine", "Lara", "Lilli", "Luna",
    "Maja", "Mathilda", "Mira", "Nathalie", "Pia", "Ronja", "Stella", "Valentina",
    "Vanessa", "Veronika", "Victoria", "Vivien", "Antonia", "Celina", "Celine",
    "Chiara", "Emely", "Emily", "Ina", "Jessica", "Judith", "Larissa", "Lina",
    "Linda", "Louisa", "Luise", "Lynn", "Magdalena", "Michelle", "Milena", "Nadja",
    "Nele", "Patricia", "Rebecca", "Regina", "Sabrina", "Sara", "Selina", "Sina",
    "Svenja", "Tamara", "Tatjana", "Tina", "Verena", "Xenia", "Yvette", "Leonie",
    "Hannah", "Sophia", "Isabell", "Isabel", "Manuela", "Monika", "Ilona", "Gabi",
    "Lieselotte", "Walburga", "Andrea", "Sonia", "Elisa", "Fiona", "Ida", "Nina",
    "Romy", "Tara", "Wanda", "Yasmin", "Zlata", "Adele", "Brunhilde", "Gudrun",
    "Hedwig", "Irmgard", "Karola", "Liesel", "Mechthild", "Sigrid", "Traudel",
    "Hilde", "Elfriede", "Anneliese", "Rosemarie", "Marlene", "Ingrid", "Gerta"
]

MALE_NAMES = [
    "Thomas", "Michael", "Andreas", "Stefan", "Markus", "Christian", "Daniel",
    "Martin", "Klaus", "Peter", "Hans", "Wolfgang", "Juergen", "Frank", "Matthias",
    "Tobias", "Sebastian", "Alexander", "Johannes", "Florian", "David", "Felix",
    "Max", "Philipp", "Tim", "Jan", "Lukas", "Jonas", "Niklas", "Simon", "Leon",
    "Paul", "Noah", "Elias", "Julian", "Fabian", "Dominik", "Patrick", "Robert",
    "Marco", "Lars", "Sven", "Uwe", "Ralf", "Holger", "Dirk", "Carsten", "Torsten",
    "Bernd", "Karl", "Fritz", "Heinrich", "Ernst", "Otto", "Wilhelm", "Gerhard",
    "Helmut", "Herbert", "Horst", "Kurt", "Manfred", "Rudolf", "Walter", "Werner",
    "Guenther", "Heinz", "Josef", "Armin", "Axel", "Benedikt", "Christoph", "Clemens",
    "Dennis", "Dieter", "Egon", "Emil", "Erich", "Erik", "Finn", "Georg", "Harald",
    "Jochen", "Konrad", "Lennart", "Lorenz", "Ludwig", "Lutz", "Marvin", "Moritz",
    "Nils", "Norman", "Oliver", "Pascal", "Rainer", "Reinhard", "Rolf", "Stephan",
    "Udo", "Ulrich", "Viktor", "Volker", "Willi", "Yannick", "Aaron", "Adam",
    "Adrian", "Albert", "Alfred", "Anton", "Arthur", "August", "Bastian", "Bruno",
    "Cedric", "Christof", "Detlef", "Eduard", "Ferdinand", "Friedrich", "Gregor",
    "Hannes", "Hugo", "Igor", "Ivan", "Joachim", "Joerg", "Kevin", "Kilian",
    "Leander", "Leo", "Leopold", "Marius", "Maurice", "Maximilian", "Nico", "Nicolas",
    "Oskar", "Piet", "Raphael", "Richard", "Roland", "Roman", "Samuel", "Theo",
    "Till", "Timo", "Tom", "Tristan", "Valentin", "Vincent", "Yannik", "Bjorn",
    "Bjoern", "Henrik", "Finn", "Lasse", "Thilo", "Thorsten", "Tobias", "Ulf",
    "Hartmut", "Dietmar", "Gerd", "Guenter", "Reiner", "Sigmund", "Volkmar",
    "Wolfram", "Xaver", "Zoran", "Mustafa", "Mohammed", "Ahmed", "Abdullah",
    "Omar", "Hassan", "Ali", "Ibrahim", "Mehmet", "Yusuf", "Karim", "Tariq",
    "Khalid", "Omer", "Sinan", "Emre", "Murat", "Kemal", "Can", "Luca", "Finn",
    "Samir", "Sameer", "Raj", "Rahul", "Arjun", "Vikram", "Nikhil", "Tarun"
]

AMBIGUOUS_NAMES = [
    "Sascha", "Kim", "Robin", "Toni", "Alex", "Chris", "Sandy", "Pat", "Terry",
    "Dana", "Charlie", "Jamie", "Jordan", "Casey", "Morgan", "Quinn", "Avery",
    "Bailey", "Riley", "Jessie", "Nicky", "Sam", "Frankie", "Lee", "Shane",
    "Drew", "Blake", "Logan", "Jean", "Dominique", "Michi", "Niki", "Sasha",
    "Rene", "Claude", "Jan", "Kai", "Michel", "Jule", "Andy", "Mikael"
]

FEMALE_NAMES_INTL = [
    "Fatima", "Aisha", "Zainab", "Mariam", "Nadia", "Layla", "Yasmin", "Leila",
    "Dilan", "Elif", "Ayse", "Zeynep", "Esra", "Merve", "Selin", "Irem",
    "Olga", "Natasha", "Oksana", "Svetlana", "Irina", "Tatiana", "Ekaterina",
    "Giulia", "Alessia", "Francesca", "Isabella", "Sofia", "Camila", "Gabriela",
    "Priya", "Ananya", "Divya", "Kavya", "Pooja", "Sneha", "Swati", "Trisha",
    "Noa", "Maya", "Lior", "Shira", "Ayala", "Noa", "Inbar", "Gal", "Michal"
]

MALE_NAMES_INTL = [
    "Dmitri", "Sergei", "Alexei", "Nikolai", "Andrei", "Pavel", "Vladimir",
    "Bogdan", "Mihai", "Ionut", "Pedro", "Carlos", "Miguel", "Juan", "Diego",
    "Luis", "Manuel", "Rafael", "Amir", "Reza", "Darius", "Cyrus", "Kian",
    "Aryan", "Rohan", "Abdullah", "Khalid", "Yilmaz", "Oguz", "Burak", "Cem",
    "Deniz", "Ercan", "Fatih", "Gokhan", "Hakan", "Ilhan", "Kaan", "Levent"
]

SURNAMES = [
    "mueller", "schmidt", "schneider", "fischer", "weber", "meyer", "wagner",
    "becker", "schulz", "hoffmann", "schaefer", "koch", "bauer", "richter",
    "klein", "wolf", "schroeder", "neumann", "schwarz", "zimmermann", "braun",
    "krueger", "hofmann", "hartmann", "lange", "schmitt", "werner", "schmitz",
    "krause", "meier", "lehmann", "schmid", "schulze", "maier", "koehler",
    "herrmann", "koenig", "walter", "mayer", "huber", "kaiser", "fuchs",
    "peters", "lang", "scholz", "moeller", "weiss", "jung", "hahn", "schubert",
    "vogel", "friedrich", "keller", "guenther", "frank", "berger", "winkler",
    "roth", "beck", "lorenz", "baumann", "franke", "albrecht", "schumann",
    "simon", "busch", "engel", "vogt", "sauer", "arnold", "brandt", "haas",
    "sommer", "gruber", "hofer", "pichler", "steiner", "moser", "wimmer",
    "eder", "stadler", "leitner", "brunner", "reiter", "koller", "egger",
    "rosenbaum", "goldberg", "steinberg", "blum", "feldmann", "kaufmann",
    "bachmann", "bergmann", "staudinger", "reimann", "hoffmeister", "demir",
    "yilmaz", "kaya", "celik", "aydin", "arslan", "koc", "ozturk", "sahin",
    "petrov", "rossi", "garcia", "martin", "santos", "oliveira", "rodriguez",
    "fink", "weigel", "bauer", "seidel", "geiger", "pfeiffer", "lenz", "haas"
]

DOMAINS = [
    "gmail.com", "web.de", "gmx.de", "outlook.com", "yahoo.de", "hotmail.de",
    "t-online.de", "freenet.de", "icloud.com", "live.de", "outlook.de",
    "gmx.net", "posteo.de", "protonmail.com", "mailbox.org",
    "berlin-data.de", "smartfactory.de", "alpha-consult.de", "innovatech.de",
    "nordstern.io", "pixelwerk.de", "musterfirma.de", "stadtlabor.de",
    "techsolutions.de", "digitalwerk.de", "mediafabrik.de", "softwarehaus.de"
]

FUNCTIONAL_PREFIXES = [
    "info", "kontakt", "service", "support", "admin", "office", "mail",
    "post", "anfrage", "bestellung", "buchhaltung", "vertrieb", "marketing",
    "hr", "sekretariat", "empfang", "redaktion", "presse", "verwaltung",
    "noreply", "no-reply", "newsletter", "team", "hilfe", "technik",
    "datenschutz", "feedback", "bewerbung", "jobs", "karriere", "einkauf",
    "lager", "werkstatt", "produktion", "qualitaet", "it", "web", "shop"
]

SEPARATORS = [".", "-", "_", ""]
NOISE_CHARS = list("0123456789") + list("0123456789") + ["_", "-", "."]


def rand_num(max_val=999):
    return str(random.randint(1, max_val))


def rand_year():
    return str(random.randint(1960, 2005))


def rand_domain():
    return random.choice(DOMAINS)


def add_noise(s):
    sep = random.choice(["", "_", "-", "."])
    noise = "".join(random.choices(string.digits, k=random.randint(1, 4)))
    return s + sep + noise


def make_sep(a, b):
    return a + random.choice(SEPARATORS) + b


def generate_email(email_type, gender):
    domain = rand_domain()

    if email_type == "clear_firstname_lastname":
        if gender == "woman":
            first = random.choice(FEMALE_NAMES).lower()
        elif gender == "man":
            first = random.choice(MALE_NAMES).lower()
        else:
            first = random.choice(AMBIGUOUS_NAMES).lower()
        last = random.choice(SURNAMES)
        local = make_sep(first, last)

    elif email_type == "firstname_number":
        if gender == "woman":
            first = random.choice(FEMALE_NAMES).lower()
        elif gender == "man":
            first = random.choice(MALE_NAMES).lower()
        else:
            first = random.choice(AMBIGUOUS_NAMES).lower()
        local = first + random.choice(["", "_", ""]) + rand_num(9999)

    elif email_type == "initial_lastname":
        first = random.choice(FEMALE_NAMES + MALE_NAMES + AMBIGUOUS_NAMES)
        last = random.choice(SURNAMES)
        local = first[0].lower() + random.choice([".", ""]) + last

    elif email_type == "firstname_lastname_noise":
        if gender == "woman":
            first = random.choice(FEMALE_NAMES).lower()
        elif gender == "man":
            first = random.choice(MALE_NAMES).lower()
        else:
            first = random.choice(AMBIGUOUS_NAMES).lower()
        last = random.choice(SURNAMES)
        local = make_sep(first, last) + random.choice(["", "_", "-"]) + rand_num(999)

    elif email_type == "lastname_number":
        last = random.choice(SURNAMES)
        local = last + rand_num(999)

    elif email_type == "abbreviation":
        first = random.choice(FEMALE_NAMES + MALE_NAMES + AMBIGUOUS_NAMES)
        last = random.choice(SURNAMES)
        local = first[:2].lower() + last[:3].lower()

    elif email_type == "functional_address":
        prefix = random.choice(FUNCTIONAL_PREFIXES)
        local = prefix + random.choice(["", rand_num(99)])

    elif email_type == "hyphenated_multiname":
        if gender == "woman":
            first1 = random.choice(FEMALE_NAMES).lower()
            first2 = random.choice(FEMALE_NAMES).lower()
        else:
            first1 = random.choice(MALE_NAMES).lower()
            first2 = random.choice(MALE_NAMES).lower()
        last = random.choice(SURNAMES)
        local = first1 + "-" + first2 + random.choice(["", ".", "_"]) + last

    elif email_type == "surname_only":
        last = random.choice(SURNAMES)
        local = last + random.choice(["", rand_num(99)])

    elif email_type == "cross_gender_name":
        first = random.choice(AMBIGUOUS_NAMES).lower()
        last = random.choice(SURNAMES)
        local = make_sep(first, last)

    elif email_type == "international_name":
        if gender == "woman":
            first = random.choice(FEMALE_NAMES_INTL).lower()
        else:
            first = random.choice(MALE_NAMES_INTL).lower()
        last = random.choice(SURNAMES)
        local = make_sep(first, last)

    elif email_type == "transliterated_name":
        UMLAUT_NAMES_M = ["bjoern", "juergen", "guenther", "uenal", "goekhan", "oezkan", "oeztuerk"]
        UMLAUT_NAMES_F = ["baerbel", "goethe", "roesel", "haeussler", "zuerich"]
        if gender == "woman":
            first = random.choice(UMLAUT_NAMES_F)
        else:
            first = random.choice(UMLAUT_NAMES_M)
        last = random.choice(SURNAMES)
        local = make_sep(first, last)

    elif email_type == "nickname":
        NICKNAMES_M = ["der_tiger", "blitz99", "nighthawk", "ironman", "darkwolf", "speedster",
                       "bigboss", "thunder", "stormrider", "coolguy", "matze", "hansi", "sebi",
                       "flo_93", "der_max", "phili", "tobi_cool", "alex_g", "stefan88"]
        NICKNAMES_F = ["sunny_girl", "pinky", "lena_sweet", "princess", "butterfly", "moonlight",
                       "strawberry", "glitter", "sophie_s", "annabell", "lara_la", "nina_x",
                       "emmi_j", "juli_chen", "sandraS", "little_star", "rosy", "laury"]
        NICKNAMES_U = ["x_gaming", "user1234", "anonymous_99", "the_one", "zz_top",
                       "keyboard_warrior", "silent_q", "abc_xyz", "hello_world", "test123"]
        if gender == "woman":
            local = random.choice(NICKNAMES_F)
        elif gender == "man":
            local = random.choice(NICKNAMES_M)
        else:
            local = random.choice(NICKNAMES_U)

    else:
        local = "user" + rand_num(9999)

    local = local.replace(" ", "").lower()
    return f"{local}@{domain}"


def build_rows(target_counts):
    rows = []
    for (email_type, gender), count in target_counts.items():
        for _ in range(count):
            email = generate_email(email_type, gender)
            rows.append({"email": email, "gender": gender, "email_type": email_type})
    random.shuffle(rows)
    return rows


TRAIN_DISTRIBUTION = {
    ("clear_firstname_lastname", "woman"):      12000,
    ("clear_firstname_lastname", "man"):        12000,
    ("firstname_number", "woman"):               5000,
    ("firstname_number", "man"):                 5000,
    ("firstname_number", "unknown"):             2000,
    ("initial_lastname", "unknown"):             9000,
    ("firstname_lastname_noise", "woman"):       4500,
    ("firstname_lastname_noise", "man"):         4500,
    ("lastname_number", "unknown"):              6000,
    ("abbreviation", "unknown"):                 7000,
    ("functional_address", "unknown"):           7000,
    ("hyphenated_multiname", "woman"):           2500,
    ("hyphenated_multiname", "man"):             2500,
    ("surname_only", "unknown"):                 5000,
    ("cross_gender_name", "unknown"):            4000,
    ("international_name", "woman"):             2000,
    ("international_name", "man"):               2000,
    ("transliterated_name", "woman"):            1000,
    ("transliterated_name", "man"):              1000,
    ("nickname", "woman"):                       2000,
    ("nickname", "man"):                         2000,
    ("nickname", "unknown"):                     2000,
}

TEST_DISTRIBUTION = {
    ("cross_gender_name", "unknown"):            100,
    ("international_name", "woman"):              60,
    ("international_name", "man"):                60,
    ("nickname", "woman"):                        50,
    ("nickname", "man"):                          50,
    ("nickname", "unknown"):                      50,
    ("clear_firstname_lastname", "woman"):        80,
    ("clear_firstname_lastname", "man"):          80,
    ("firstname_lastname_noise", "woman"):        40,
    ("firstname_lastname_noise", "man"):          40,
    ("firstname_number", "woman"):                30,
    ("firstname_number", "man"):                  30,
    ("firstname_number", "unknown"):              20,
    ("initial_lastname", "unknown"):               60,
    ("functional_address", "unknown"):            60,
    ("abbreviation", "unknown"):                  50,
    ("transliterated_name", "woman"):             20,
    ("transliterated_name", "man"):               20,
    ("hyphenated_multiname", "woman"):            20,
    ("hyphenated_multiname", "man"):              20,
    ("lastname_number", "unknown"):               30,
    ("surname_only", "unknown"):                  30,
}


def main():
    os.makedirs("data", exist_ok=True)

    random.seed(42)
    train_rows = build_rows(TRAIN_DISTRIBUTION)
    df_train = pd.DataFrame(train_rows)
    df_train.insert(0, "id", range(1, len(df_train) + 1))
    df_train.to_csv("data/train.csv", index=False, encoding="utf-8")
    print(f"train.csv: {len(df_train)} rows")
    print(df_train["gender"].value_counts())
    print()

    random.seed(99)
    test_rows = build_rows(TEST_DISTRIBUTION)
    df_test = pd.DataFrame(test_rows)
    df_test.insert(0, "id", range(1, len(df_test) + 1))
    df_test.to_csv("data/test.csv", index=False, encoding="utf-8")
    print(f"test.csv: {len(df_test)} rows")
    print(df_test["gender"].value_counts())


if __name__ == "__main__":
    main()

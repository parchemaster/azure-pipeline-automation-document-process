import json
import os
import re
import numpy as np
import pandas as pd
import traceback
import cv2
# import pypdfium2 as pdfium

from pdf2image import convert_from_path
import pytesseract
import datetime
import shutil
# import win32security
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from configparser import ConfigParser

# poppler_path = '/usr/local/bin'
# poppler_path = 'src/poppler-0.67.0/bin'


config_object = ConfigParser()
config_object.read("src/config.ini")
counter = (int)(config_object["OTHER"]["FileCounter"])


def moveToErrorDir(inFilepath, owner):
    global error_messages
    # createEmail(owner, config_object["EMAIL"]["EmailErrorText"].replace("missing", ','.join(error_messages)), inFilepath, "NONO")
    print("error")
    shutil.move(inFilepath, os.path.join(
        config_object["DIRECTION"]["ErrorPath"], os.path.basename(inFilepath)))


def moveToProcessedDir(inFilepath, CSVname, owner, csvFilePath):
    # createEmail(owner, config_object["EMAIL"]["EmailProcessedText"], inFilepath, csvFilePath)
    shutil.move(inFilepath, os.path.join(config_object["DIRECTION"]["ProcessedPath"], os.path.basename(
        inFilepath).split(".")[-2] + "_" + CSVname + ".pdf"))


def createCSVName():
    year = datetime.date.today().year
    month = datetime.date.today().month
    day = datetime.date.today().day
    global counter
    stringYear = str(year).removeprefix("20")
    name = "HP" + stringYear + \
        str(month).rjust(2, '0') + str(day).rjust(2, '0') + \
        str(counter).rjust(2, '0') + ".csv"
    counter += 1
    return name


def write_csv_PO(csvName_path):
    with open(csvName_path, 'w', encoding="ANSI", errors="ignore") as f:
        rows = []
        rows.append(
            ";;Hromadný platobný príkaz pre SEPA úhrady;;;;;;;;;;;;;;;;;;;;;;")
        rows.append(
            ";Dátum odpísania penažných prostriedkov z úctu:;;;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";" + date + ";;;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";IBAN platitela;" + ibanPlatitel +
                    ";názov úctu platitela;;;;;;;;;;;;;;;;;;;;;")
        rows.append(
            ";IBAN príjemcu;názov úctu príjemcu;mena;ciastka;VS;KS;SS;identifikácia platby;;;;;;;;;;;;;;;;")
        rows.append(";" + ibanPrijemcu + ";" + nazovUctuPrij + ";EUR;" +
                    amount + ";" + symbol + ";;;" + info + ";;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;EUR;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;;;;;;;;;;;;;;;;;;;;;;")
        rows.append(";;;;;;;;;;;;;;;;;;;;;;;;")
        for i in range(23):
            f.write(rows[i] + "\n")

        f.close()


def checkInDirPDF():
    # print(config_object["DIRECTION"]["InPath"])
    pdfs = []
    for filename in os.listdir(config_object["DIRECTION"]["InPath"]):
        split_filename = os.path.splitext(filename)
        if (split_filename[1].lower() == ".pdf"):
            pdfs.append(os.path.join(
                config_object["DIRECTION"]["InPath"], filename))
    return pdfs


def readOwner(file_path):
    sd = win32security.GetFileSecurity(
        file_path, win32security.OWNER_SECURITY_INFORMATION)
    owner_sid = sd.GetSecurityDescriptorOwner()
    name, domain, type = win32security.LookupAccountSid(None, owner_sid)
    print(f'The owner of {file_path} is {name}\\{domain}\n')
    return name


def findEmployee(owner):
    data = pd.read_csv('src/employeeData.csv')
    for index, row in data.iterrows():
        if owner in row[2]:
            return(row['email'])


def createEmail(owner, text, pdf_path, csv_path):
    receiver = findEmployee(owner)

    message = MIMEMultipart("alternative")
    message["Subject"] = "Spracovanie platobného príkazu"
    message["From"] = config_object["EMAIL"]["Sender"]
    message["To"] = receiver

    newText = str(text)
    try:
        with open(pdf_path, "rb") as f:
            file_dir1, file_name1 = os.path.split(pdf_path)
            attach1 = MIMEApplication(f.read(), _subtype="pdf")
            attach1.add_header('content-disposition',
                               'attachment', filename=file_name1)
            text = text.replace("prikaz", "prikaz " + file_name1)
            message.attach(attach1)

        if (csv_path != "NONO"):
            with open(csv_path, "rb") as f:
                file_dir1, file_name2 = os.path.split(csv_path)
                attach2 = MIMEApplication(f.read(), _subtype="csv")
                attach2.add_header('content-disposition',
                                   'attachment', filename=file_name2)
                text = text.replace("csv", file_name2)
                message.attach(attach2)
        stringText = str(text).replace("\\n", "\n")
        part1 = MIMEText(stringText, "plain")
        message.attach(part1)

        with smtplib.SMTP("10.0.0.172") as server:
            server.sendmail(
                config_object["EMAIL"]["Sender"], receiver, message.as_string())
            print("Successfully sent email\n")
    except Exception as e:
        print("Error {e}: unable to send email\n")


def convertPDF(pdf_path):

    # pdffile = pdf_path
    # doc = fitz.open(pdffile)
    # zoom = 4
    # mat = fitz.Matrix(zoom, zoom)
    # count = 0
    # # Count variable is to get the number of pages in the pdf
    # for p in doc:
    #     count += 1
    # for i in range(count):
    #     val = f"src/image_for_text_detection.jpg"
    #     page = doc.load_page(i)
    #     pix = page.get_pixmap(matrix=mat)
    #     pix.save(val)
    # doc.close()

    images = convert_from_path(pdf_path)
    images[0].save("src/image_for_text_detection.jpg", 'JPEG')


def crop_invoice(file_name):
    img = cv2.imread("src/image_for_text_detection.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    black_lines = []
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            black_lines.append(contour)

    biggest_crop = None
    biggest_area = 0
    second_biggest_crop = None
    second_biggest_area = 0

    for line in black_lines:
        x, y, w, h = cv2.boundingRect(line)
        crop_img = img[y:y + h, x:x + w]
        area = w * h
        if area > biggest_area:
            second_biggest_area = biggest_area
            second_biggest_crop = biggest_crop
            biggest_area = area
            biggest_crop = crop_img
        elif area > second_biggest_area:
            second_biggest_area = area
            second_biggest_crop = crop_img
    new_size = (700, 900)
    cv2_img = cv2.resize(second_biggest_crop, new_size)
    # cv2.imshow("croped image", cv2_img)
    # cv2.waitKey(0)
    cv2.imwrite('src/cropped_image.png', second_biggest_crop)


def deskew(im, max_skew=10):
    height, width, channels = im.shape

    # Create a grayscale image and denoise it
    im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gs = cv2.fastNlMeansDenoising(im_gs, h=3)

    # Create an inverted B&W copy using Otsu (automatic) thresholding
    im_bw = cv2.threshold(
        im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv2.HoughLinesP(
        im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )

    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum(
        [abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(
            angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        # Insufficient data to deskew
        return im

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))

    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    im = cv2.warpAffine(im, M, (width, height),
                        borderMode=cv2.BORDER_REPLICATE)
    return im


def remove_border(image_path):
    image = cv2.imread(image_path)
    # print(image)
    image = deskew(image)
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    new_size = (700, 900)
    cv2_img = cv2.resize(gray, new_size)
    # cv2.imshow("gray", cv2_img)
    # cv2.waitKey(0)

    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cv2_img = cv2.resize(thresh, new_size)
    # cv2.imshow("thresh", cv2_img)
    # cv2.waitKey(0)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (20, 1))  # 40
    remove_horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(
        remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))  # 20
    remove_vertical = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(
        remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

    cv2_img = cv2.resize(result, new_size)
    # cv2.imshow('result', cv2_img)
    cv2.imwrite(r'src/result.png', result)
    # cv2.waitKey()


def write_possition_and_detect(name, im2, x, y, w, h, percentage_x, percentage_y, percentage_width, percentage_height, config):
    # rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped = im2[y:y + h, x:x + w]
    text = pytesseract.image_to_string(cropped, lang='eng+slk', config=config)
    match = re.search(r'[a-zA-Z]+', text) or re.search(r'[0-9]+', text)
    if match:
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = text.replace("EUR", "").replace("€", "C").strip()
        cv2.putText(im2, text, (x + 0, y + 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 1, cv2.LINE_AA)
        with open(r"src/recognized.txt", "a", encoding="utf-8") as f:
            f.write(name + text)
            # print(name + text + " -X%: " + str(format(percentage_x, '.2f') + "  Y%: " + str(format(percentage_y, '.2f')
            #                                                                                 ) + " -W%: " + str(format(percentage_width, '.2f')) + " -H%: " + str(format(percentage_height, '.2f'))))
            f.write("\n")
        f.close
        return text
    else:
        return ""


def remove_duplicates_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    lines = list(set(lines))
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(lines)


def removeBlackDots():
    image = cv2.imread(r'src/result.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < 10:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    result = 255 - thresh

    # Show the result
    new_size = (700, 900)
    cv2_img = cv2.resize(result, new_size)
    # cv2.imshow('Result1', cv2_img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(r'src/result.png', result)


def create_json(file_name):
    data = {
        "date": date, "iban_platitel": ibanPlatitel, "iban_prijemcu": ibanPrijemcu,
        "nazov_uctu": nazovUctuPrij, "suma": amount, "symbol": symbol, "info": info
    }
    df = pd.DataFrame(data, index=[0])

    # Write the dataframe to a JSON file
    # df.to_json(r"data\json\\" + file_name + ".json", orient='records', indent=4)


def main_test(json_path, file_name):
    if(json_path == "src/cordinates_percentage.json"):
        crop_invoice(file_name)
        remove_border("src/cropped_image.png")
        changed_y = 4
    elif(json_path == "src/cordinates_Interny.json"):
        remove_border("src/image_for_text_detection.jpg")
        changed_y = 1
    else:
        remove_border("src/image_for_text_detection.jpg")
        changed_y = 4

    img = cv2.imread("src/result.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()
    global ibanPlatitel
    global ibanPrijemcu
    global nazovUctuPrij
    global symbol
    global amount
    global info
    global date

    now = datetime.date.today()

    ibanPlatitel = ""
    ibanPrijemcu = ""
    nazovUctuPrij = ""
    symbol = ""
    amount = ""
    info = ""
    date = now.strftime('%d.%m.%Y')
    ibans = []

    file = open("src/recognized.txt", "w+")
    file.write("")
    file.close()

    df = pd.read_json(json_path)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        width, high = im2.shape[1], im2.shape[0]
        percentage_y = ((y + (h / 2)) / high) * 100
        percentage_x = ((x + (w / 2)) / width) * 100
        percentage_width = (w / width) * 100
        percentage_height = (h / high) * 100

        models = df['models']
        for model in models:
            # if(
            #     (percentage_x == model['date']['percentage_x'] or percentage_x >= model['date']['percentage_x'] - 4 and percentage_x <= model['date']['percentage_x'] + 4)
            #     and
            #     (percentage_y == model['date']['percentage_y'] or percentage_y >= model['date']['percentage_y'] - 4 and percentage_y <= model['date']['percentage_y'] + 4)
            #     and
            #     (percentage_width == model['date']['percentage_width'] or percentage_width >= model['date']['percentage_width'] - 10 and percentage_width <= model['date']['percentage_width'] + 10)
            #     and
            #     (percentage_height == model['date']['percentage_height'] or percentage_height >= model['date']['percentage_height'] - 2 and percentage_height <= model['date']['percentage_height'] + 2)
            # ):
            #     date = write_possition_and_detect("date: ", im2, x, y, w, h, percentage_x, percentage_y, percentage_width, percentage_height, '--psm 6 -c tessedit_char_whitelist=0123456789/.-')
            #     break

            if(
                json_path != "cordinates_Interny.json"
                and
                (percentage_x == model['iban1']['percentage_x'] or percentage_x >= model['iban1']
                 ['percentage_x'] - 4 and percentage_x <= model['iban1']['percentage_x'] + 4)
                and
                (percentage_y == model['iban1']['percentage_y'] or percentage_y >= model['iban1']
                 ['percentage_y'] - 4 and percentage_y <= model['iban1']['percentage_y'] + 4)
                and
                (percentage_width == model['iban1']['percentage_width'] or percentage_width >= model['iban1']
                 ['percentage_width'] - 5 and percentage_width <= model['iban1']['percentage_width'] + 5)
                and
                (percentage_height == model['iban1']['percentage_height'] or percentage_height >= model['iban1']
                 ['percentage_height'] - 2 and percentage_height <= model['iban1']['percentage_height'] + 2)
            ):
                ibanPlatitel = write_possition_and_detect(
                    "iban1: ", im2, x, y, w, h, percentage_x, percentage_y, percentage_width, percentage_height, "").replace(" ", "").replace("$", "S").upper()
                break

            elif(
                (percentage_x == model['iban2']['percentage_x'] or percentage_x >= model['iban2']
                 ['percentage_x'] - 4 and percentage_x <= model['iban2']['percentage_x'] + 4)
                and
                (percentage_y == model['iban2']['percentage_y'] or percentage_y >= model['iban2']
                 ['percentage_y'] - 4 and percentage_y <= model['iban2']['percentage_y'] + 4)
                and
                (percentage_width == model['iban2']['percentage_width'] or percentage_width >= model['iban2']
                 ['percentage_width'] - 5 and percentage_width <= model['iban2']['percentage_width'] + 5)
                and
                (percentage_height == model['iban2']['percentage_height'] or percentage_height >= model['iban2']
                 ['percentage_height'] - 2 and percentage_height <= model['iban2']['percentage_height'] + 2)
            ):
                ibanPrijemcu = write_possition_and_detect(
                    "iban2: ", im2, x, y, w, h, percentage_x, percentage_y, percentage_width, percentage_height, "").replace(" ", "").replace("$", "S").upper()
                ibans.append(ibanPrijemcu)
                break

            elif(
                (percentage_x == model['name']['percentage_x'] or percentage_x >= model['name']
                 ['percentage_x'] - 4 and percentage_x <= model['name']['percentage_x'] + 4)
                and
                (percentage_y == model['name']['percentage_y'] or percentage_y >= model['name']
                 ['percentage_y'] - changed_y and percentage_y <= model['name']['percentage_y'] + changed_y)
                and
                (percentage_width == model['name']['percentage_width'] or percentage_width >= model['name']
                 ['percentage_width'] - 70 and percentage_width <= model['name']['percentage_width'] + 70)
                and
                (percentage_height == model['name']['percentage_height'] or percentage_height >= model['name']
                 ['percentage_height'] - 5 and percentage_height <= model['name']['percentage_height'] + 5)
            ):
                nazovUctuPrij = write_possition_and_detect(
                    "name: ", im2, x, y, w, h, percentage_x, percentage_y, percentage_width, percentage_height, '--psm 11')
                if ("BRANDS" in nazovUctuPrij.upper()):
                    nazovUctuPrij = nazovUctuPrij.replace(
                        "1", "I").replace("G", "Q")
                if ("CHEMOLA" in nazovUctuPrij.upper()):
                    nazovUctuPrij = nazovUctuPrij.replace("as.", "a.s.")
                break

            elif(
                (percentage_x == model['suma']['percentage_x'] or percentage_x >= model['suma']
                 ['percentage_x'] - 4 and percentage_x <= model['suma']['percentage_x'] + 4)
                and
                (percentage_y == model['suma']['percentage_y'] or percentage_y >= model['suma']
                 ['percentage_y'] - changed_y and percentage_y <= model['suma']['percentage_y'] + changed_y)
                and
                (percentage_width == model['suma']['percentage_width'] or percentage_width >= model['suma']
                 ['percentage_width'] - 10 and percentage_width <= model['suma']['percentage_width'] + 10)
                and
                (percentage_height == model['suma']['percentage_height'] or percentage_height >= model['suma']
                 ['percentage_height'] - 2 and percentage_height <= model['suma']['percentage_height'] + 2)
            ):
                amount = write_possition_and_detect(
                    "suma: ", im2, x, y, w, h, percentage_x, percentage_y, percentage_width, percentage_height, "--psm 6")
                break

            elif(
                (percentage_x == model['symbol']['percentage_x'] or percentage_x >= model['symbol']
                 ['percentage_x'] - 4 and percentage_x <= model['symbol']['percentage_x'] + 4)
                and
                (percentage_y == model['symbol']['percentage_y'] or percentage_y >= model['symbol']
                 ['percentage_y'] - changed_y and percentage_y <= model['symbol']['percentage_y'] + changed_y)
                and
                (percentage_width == model['symbol']['percentage_width'] or percentage_width >= model['symbol']
                 ['percentage_width'] - 20 and percentage_width <= model['symbol']['percentage_width'] + 20)
                and
                (percentage_height == model['symbol']['percentage_height'] or percentage_height >= model['symbol']
                 ['percentage_height'] - 2 and percentage_height <= model['symbol']['percentage_height'] + 2)
            ):
                symbol = write_possition_and_detect(
                    "symbol: ", im2, x, y, w, h, percentage_x, percentage_y, percentage_width, percentage_height, '--psm 10')
                break

            elif(
                (percentage_x == model['info']['percentage_x'] or percentage_x >= model['info']
                 ['percentage_x'] - 5 and percentage_x <= model['info']['percentage_x'] + 5)
                and
                (percentage_y == model['info']['percentage_y'] or percentage_y >= model['info']
                 ['percentage_y'] - 4 and percentage_y <= model['info']['percentage_y'] + 4)
                and
                (percentage_width == model['info']['percentage_width'] or percentage_width >= model['info']
                 ['percentage_width'] - 70 and percentage_width <= model['info']['percentage_width'] + 70)
                and
                (percentage_height == model['info']['percentage_height'] or percentage_height >= model['info']
                 ['percentage_height'] - 2 and percentage_height <= model['info']['percentage_height'] + 5)
            ):
                info = write_possition_and_detect(
                    "info: ", im2, x, y, w, h, percentage_x, percentage_y, percentage_width, percentage_height, "")
                break
        # write_possition_and_detect("info: ", im2, x, y, w, h, percentage_x, percentage_y, percentage_width, percentage_height, "")
    if (len(ibans) == 2 and json_path == "src/cordinates_Interny.json"):
        ibanPrijemcu = ibans[0]
        ibanPlatitel = ibans[1]

    print("iban sender: " + ibanPlatitel)
    print("iban receiver: " + ibanPrijemcu)
    print("receiver name " + nazovUctuPrij)
    print("money amount: " + amount)

    # print("")

    remove_duplicates_from_file("src/recognized.txt")
    cv2.imwrite('src/framed_result.png', im2)
    new_size = (700, 900)
    cv2_img = cv2.resize(im2, new_size)
    # cv2.imshow(file_name, cv2_img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    create_json(file_name)


def is_Interny_doc_type(file):
    remove_border(file)
    image = cv2.imread(file)
    text = pytesseract.image_to_string(image)
    if "PRIKAZ K UHRADE" in text.upper():
        return True
    return False


def testing_SEPO(file_name):
    json_path = "src/data\json\\" + file_name + ".json"

    if os.path.exists(json_path):
        df = pd.read_json(json_path)
        for index, row in df.iterrows():
            print(str(row['iban_platitel']) + "  " + str(ibanPlatitel))
            print(str(row['iban_prijemcu']) + "  " + str(ibanPrijemcu))
            print(str(row['nazov_uctu']) + "  " + str(nazovUctuPrij))
            print(str(row['suma']) + "  " + amount)
            print(str(row['symbol']) + "  " + str(symbol))
            print(str(row['info']) + "  " + str(info))
            if (str(row['iban_platitel']) == ibanPlatitel and str(row['iban_prijemcu']) == ibanPrijemcu and str(row['nazov_uctu']) == nazovUctuPrij
                    and str(row['suma']) == amount and str(row['symbol']) == str(symbol)):
                shutil.move(file, r"src/data/good_pdf")
                print("File: " + file_name + " was good tested")
            else:
                shutil.move(file, r"src/data/bad_pdf")
                print("File: " + file_name + " was bad tested")
    else:
        shutil.move(file, r"src/data/bad_pdf")
        print("There is no json for " + file_name)


def add_error_message(item_control, text_message):
    global error_messages
    if(item_control == ""):
        error_messages.append(text_message)


if __name__ == '__main__':
    try:
        files_to_process = checkInDirPDF()
        for file in files_to_process:
            file_name = os.path.splitext(os.path.basename(file))[0]
            # text_file_path = unidecode(os.path.join(config_object["DIRECTION"]["TextPath"], file_name + ".txt"))
            # owner = readOwner(file)
            owner = "test"
            error_messages = []
            # convertPDF(file)

            if (is_Interny_doc_type("src/image_for_text_detection.jpg")):
                main_test("src/cordinates_Interny.json", file_name)

                add_error_message(ibanPlatitel, "Chýba IBAN odosielateľa")
                add_error_message(ibanPrijemcu, "Chýba IBAN prijímača")
                add_error_message(symbol, "Chýba Variabilný symbol")
                add_error_message(amount, "Chýba suma")
            else:
                main_test("src/cordinates_percentage.json", file_name)
                with open("src/recognized.txt", "r", encoding="utf-8") as f:
                    file_contents = f.read()

                    if("iban1" not in file_contents or "iban2" not in file_contents or "name" not in file_contents or "suma" not in file_contents):
                        print(
                            "the first method didn't work. Program will run without cropping")
                        main_test("cordinates_percentage_v2.json", file_name)

                        add_error_message(
                            ibanPlatitel, "Chýba IBAN odosielateľa")
                        add_error_message(ibanPrijemcu, "Chýba IBAN prijímača")
                        add_error_message(
                            nazovUctuPrij, "Chýba Nazov uctu prijemca")
                        add_error_message(amount, "Chýba suma")

                f.close
            # testing_SEPO(file_name)
            # if(len(error_messages) > 0):
            #     moveToErrorDir(file, owner)
            #     break
            # else:
            #     isFile = True
            #     while(isFile == True):
            #         csvName = createCSVName()
            #         csv_name_path = os.path.join(
            #             config_object["DIRECTION"]["OutPath"], csvName)
            #         isFile = os.path.isfile(csv_name_path)
            #         if(isFile == False):
            #             csv_name_path = os.path.join(
            #                 config_object["DIRECTION"]["OutPath"] + "\\archiv", csvName)
            #             isFile = os.path.isfile(csv_name_path)

            #     csv_name_path = os.path.join(
            #         config_object["DIRECTION"]["OutPath"], csvName)
            #     write_csv_PO(csv_name_path)
            #     moveToProcessedDir(file, csvName, owner, csv_name_path)
            #     print("Prikaz bol zpracovany\n\n")

    except Exception as e:
        print(e)
        traceback.print_exc()

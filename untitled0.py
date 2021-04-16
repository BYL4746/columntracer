# -*- coding: utf-8 -*-
import newColumnTracer


c = newColumnTracer.ColumnTracer()
time = [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,
    1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,
    2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,  3.2,
    3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,  4.3,
    4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,
    5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,
    6.6,  6.7,  6.8,  6.9,  7. ,  7.1,  7.2,  7.3,  7.4,  7.5,  7.6,
    7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,  8.7,
    8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,
    9.9, 10. , 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
    11. , 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9]

# D=63
conc = [-0.05376420602334342, 6.215028491851626e-11, 5.149421999917081e-07, 0.0003219367305185017, 0.008662959091310096, 0.06473501920638558, 0.2522278489682894, 0.6735683639551571, 1.4157879206829094, 2.5316882944217745, 4.037380927536538, 5.918840618079546, 8.141202069397002, 10.657443447837222, 13.415209593269594, 16.36162633323487, 19.446403312649675, 22.623635379662687, 25.85268322065134, 29.098438853673315, 32.3312030120505, 35.52633475949008, 38.6637822245689, 41.727565760551556, 44.70525837316951, 47.587490140451614, 50.36749124857622, 53.040680391922926, 55.60430031404873, 58.057099256377754, 60.39905537325046, 62.631140311118294, 64.75511782705215, 66.77337332982988, 68.68877042813789, 70.50453087511102, 72.22413464912309, 73.85123727167226, 75.3896018130116, 76.8430433628924, 78.21538404152678, 79.51041689256503, 80.73187723577917, 81.88342026385212, 82.96860384750701, 83.99087566879379, 84.9535639363548, 85.85987105150136, 86.71286969235076, 87.51550086728909, 88.27057356059744, 88.9807656539407, 89.64862585908857, 90.2765764410416, 90.8669165478152, 91.4218259944827, 91.94336937554357, 92.43350040199377, 92.89406637825832, 93.3268127499326, 93.73338766652446, 94.11534651448399, 94.47415638507822, 94.81120044939912, 95.12778221922117, 95.42512967775384, 95.70439926873668, 95.96667973594552, 96.21299580814434, 96.44431172692866, 96.66153461685877, 96.86551769884268, 97.05706334896918, 97.23692600596071, 97.40581493116125, 97.56439682553393, 97.71329830854911, 97.8531082641225, 97.98438005893972, 98.10763363859513, 98.22335750699584, 98.33201059445183, 98.43402401979823, 98.52980275178741, 98.61972717485507, 98.70415456420945, 98.78342047502595, 98.85784005035012, 98.92770925212893, 98.99330601960177, 99.05489135909502, 99.11271036907571, 99.1669932041359, 99.217955981397, 99.26580163264775, 99.31072070535738, 99.35289211554039, 99.39248385528977, 99.42965365764267, 99.46454962129566, 99.49731079754667, 99.52806774170774, 99.55694303110451, 99.58405175165841, 99.60950195493206, 99.63339508741001, 99.65582639368353, 99.67688529511048, 99.69665574542941, 99.71521656471924, 99.732641753014, 99.74900078480388, 99.76435888558099, 99.77877729151862, 99.79231349330783, 99.80502146511382, 99.81695187955663, 99.828152309566, 99.83866741790978, 99.84853913514577]

c.fit_D(time, conc)

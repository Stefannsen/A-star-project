"""
Tema 1 - Problema decuparii
Negulescu Stefan
Grupa 232

Nota: Fisierele input1.txt si input2.txt nu se afla in folderul Input,
    deoarece daca sunt citite, programul se va bloca din cauza un exceptii
    legate de input.
input1.txt - fara solutii (input invalid)
input2.txt - start = scope
input3.txt - A*/euristica neadmisibila -> rezultat gresit
input3.txt, input4.txt - input valid, se poate ajunge in starea scop

Nota: Fisierele de output se afla in folderul Output cu denumirea de forma:
    output-nume_fisier_input.txt
"""


import copy
import numpy as np
import time
import sys
import os
from queue import PriorityQueue
t1_ida = 0
t1_gs = 1000
timeout_limit = 100000


class NodParcurgere:
    def __init__(self, info, parent, cost=0.0, h=0.0):
        self.info = info
        self.parent = parent
        self.g = cost
        self.h = h
        self.f = self.g + self.h

    def get_path(self):
        path = [self]
        nod = self
        while nod.parent is not None:
            path.insert(0, nod.parent)
            nod = nod.parent
        return path

    def show_path(self, show_cost=True, show_len=True):
        path = self.get_path()
        print()
        for i, nod in enumerate(path):
            print(i+1, ":")
            print(nod)
        if show_cost:
            print("Cost: ", self.g)
        if show_len:
            print("Length: ", len(path))
        print("===========\n")
        # return len(path)

    def show_path(self, f, show_cost=True, show_len=True):
        path = self.get_path()
        f.write("\n")
        for i, nod in enumerate(path):
            f.write(str(i+1) + ":\n")
            f.write(str(nod))
        if show_cost:
            f.write("Cost: " + str(self.g) + "  ")
        if show_len:
            f.write("Length: " + str(len(path)) + "\n")
        f.write("===========\n\n")
        # return len(path)

    def in_path(self, new_node_info):
        node_path = self
        while node_path is not None:
            if np.array_equal(new_node_info, node_path.info):
                return True
            node_path = node_path.parent
        return False

    def __repr__(self):
        sir = ""
        sir += str(self.info)
        return sir

    def __str__(self):
        sir = ""
        for row in self.info:
            sir += (str(row)) + "\n"
        sir += "g: " + str(self.g) + "  h: " + str(self.h) + "\n"
        sir += "--------------\n"
        return sir


class Graph:

    def __init__(self, file_name):
        f = open(file_name, "r")
        content = f.read().split("\n\n")
        start = content[0].split("\n")
        scope = content[1].split("\n")
        self.start = np.array([list(string) for string in start])
        self.scope = np.array([list(string) for string in scope])
        f.close()

        fr = dict()
        for row in self.scope:
            for ch in row:
                if ch in fr:
                    fr[ch] += 1
                else:
                    fr[ch] = 1
        self.fr_scope = fr

        if not self.validation(start, scope):
            raise Exception("Invalid input!")

        if np.array_equal(start, scope):
            raise Exception("States start and scope are equal!")

        if not self.verify(start):
            raise Exception("Cannot reach the scope!!")

    def validation(self, start, stop):
        """
        Verificam daca inputul este corect.
        :param start:
        :param stop:
        :return:
        """
        def check_complete_matrix(matrix):  # randuri de aceeasi lungime
            for i in range(len(matrix)-1):
                if len(matrix[i]) != len(matrix[i+1]):
                    print("The state is incomplete!")
                    return False
            return True

        def check_alphabet(matrix):     # toate componentele sunt caractere intre a si z
            for row in matrix:
                for ch in row:
                    if (not ch.isalpha()) or (len(ch) != 1):
                        print("No character!")
                        return False
            return True

        def compare_dimensions(start, stop):
            if len(stop) > len(start) or len(stop[0]) > len(start[0]):
                print("Can't get this scope from this start!")
                return False
            return True

        if check_complete_matrix(start) and check_complete_matrix(stop):
            if check_alphabet(start) and check_alphabet(stop):
                if compare_dimensions(start, stop):
                    return True
        return False

    def verify(self, matrix):
        def compare_dimensions(start, stop):
            if len(stop) == len(start) and len(stop[0]) == len(start[0]):
                if not np.array_equal(matrix, self.scope):
                    return False
            if len(stop) > len(start) or len(stop[0]) > len(start[0]):
                return False
            return True

        def characters_frequency():
            current_fr = dict()
            for row in matrix:
                for ch in row:
                    if ch in current_fr:
                        current_fr[ch] += 1
                    else:
                        current_fr[ch] = 1
            for key in self.fr_scope:
                if not(key in current_fr and self.fr_scope[key] <= current_fr[key]):
                    return False
            return True

        return compare_dimensions(matrix, self.scope) and characters_frequency()

    def test_scope(self, node):
        return np.array_equal(self.scope, node.info)

    def generate_successors(self, node, h_type):
        """
        Prin verify putem vedea daca se poate ajunge din starea curenta in scop.
        Succesorii sunt generati cu ajutorul unei functii de backtracking care genereaza toate combinarile posibile
            in functie de m si n ale matricii.

        :param node:
        :param h_type:
        :return:
        """
        global t1_gs
        successors = []
        matrix = node.info
        t1_gs = time.time()

        def bkt(k, n, sol, cnt, p, arr):
            global t1_gs
            for v in range(1 if k == 0 else sol[k - 1] + 1, n + 1):
                if round(1000 * (time.time() - t1_gs)) > timeout_limit:
                    return
                sol[k] = v
                if k == p - 1:
                    cnt += 1
                    arr.append(sol[:k + 1])
                else:
                    bkt(k + 1, n, sol, cnt, p, arr)

        def cost(m, x, axa):
            nr = len(x)
            if axa == 0:
                return len(m[0]) / nr
            elif axa == 1:
                k = 0
                for i in range(len(m) - 1):
                    for j in x:
                        if m[i][j] != m[i + 1][j]:
                            # print(m[i][j], "\n")
                            k += 1

                for i in range(len(x) - 1):
                    if nr > 1:
                        if x[i + 1] == x[i] + 1:
                            for j in range(len(m)):
                                if m[j][x[i]] != m[j][x[i + 1]]:
                                    # print(m[j][x[i]], "\n")
                                    k += 1
                return 1 + k / nr

        def aux(ccc, axa):
            global t1_gs
            n = ccc
            cnt = 0
            sol = [0] * n
            arr = []

            for p in range(1, n):
                if round(1000 * (time.time() - t1_gs)) > timeout_limit:
                    print("Timeout!(gensucc)")
                    return
                bkt(0, n, sol, cnt, p, arr)

            for i in arr:
                for j in range(len(i)):
                    i[j] -= 1

            for x in arr:
                copie = copy.deepcopy(matrix)
                copie = np.delete(copie, tuple(x), axis=axa)
                if not node.in_path(copie):
                    successors.append(NodParcurgere(np.array(copie), node, node.g + cost(matrix, x, axa),
                                                    self.get_h(copie, h_type)))

        if self.verify(matrix):
            aux(len(matrix), 0)
            aux(len(matrix[0]), 1)
            return successors
        else:
            return []

    def get_h(self, node_info, h_type="euristica banala"):
        def min_columns(matrix):
            """
            Functie care calculeaza nr-ul minim de elemente vecine pe coloane care sunt diferite

            :param matrix:
            :return:
            """
            mini = len(matrix)
            for j in range(len(matrix[0])):
                nr = 0
                for i in range(len(matrix) - 1):
                    if matrix[i][j] != matrix[i + 1][j]:
                        nr += 1
                mini = min(mini, nr)
            return mini

        if h_type == "euristica banala":
            if not np.array_equal(self.scope, node_info):
                return 1
            return 0
        elif h_type == "e2":
            """
            Adaug 1 daca mai trebuie eliminate coloane.
            Adaug nr-ul de coloane din scop / nr-ul max de linii eliminate 
            """
            nr = 0
            if np.array_equal(self.scope, node_info):
                return 0
            if len(node_info[0]) > len(self.scope[0]):
                nr += 1
            if len(node_info) > len(self.scope):
                nr += len(self.scope[0]) / (len(node_info) - len(self.scope))
            return nr
        elif h_type == "e1":
            """
            Daca nr-ul de coloane este acelasi, returneaza raportul respectiv.
            Daca nr-ul d elinii este acelasi, returneaza 1 + nr-ul minim de elemente diferite de pe o coloana,
                * nrcol / nrcol
            """
            if np.array_equal(self.scope, node_info):
                return 0
            if not self.verify(node_info):
                return 100
            if len(node_info[0]) == len(self.scope[0]):  # nr coloane egal
                if len(node_info) > len(self.scope):
                    return len(self.scope[0]) / (len(node_info) - len(self.scope))
            elif len(node_info) == len(self.scope):
                if len(node_info[0]) > len(self.scope[0]):
                    dif = len(node_info[0]) - len(self.scope[0])
                    return 1 + dif * min_columns(node_info) / dif
            else:
                return 1 + len(self.scope[0]) / (len(node_info) - len(self.scope))
        elif h_type == "neadmisibila":
            """
            Neadmisibila
            """
            nr = 0
            if np.array_equal(self.scope, node_info):
                return 0
            dif_lines = len(node_info) - len(self.scope)
            dif_columns = len(node_info[0]) - len(self.scope[0])
            if dif_lines > 0:
                nr += dif_lines
            if dif_columns > 0:
                nr += dif_columns
            return nr

        #return 0

    def __repr__(self):
        sir = ""
        for (k, v) in self.__dict__.items():
            sir += "{} = {}\n".format(k, v)
        return sir


def show_stats(maxi, total, t1, f):
    f.write("Maxim number of nodes: " + str(maxi) + "\n")
    f.write("Total number of nodes: " + str(total) + "\n")
    t2 = time.time()
    milis = round(1000 * (t2 - t1))
    f.write("Time: " + str(milis))


def ucs(gr, f, count=3):
    f.write("UCS\n")
    t1 = time.time()
    maxi = 0
    total = 1
    nod = NodParcurgere(gr.start, None, 0, gr.get_h(gr.start))
    c = PriorityQueue()
    index = 0

    c.put((nod.g, index, nod))

    while c:
        if round(1000 * (time.time() - t1)) > timeout_limit:
            print("Timeout!")
            return
        maxi = max(maxi, c.qsize())
        # print("Coada actuala: ")
        # for i in c:
        #     print(i)
        # input()
        if not c.qsize():
            print("Nicio solutie!")
            return
        node = c.get()[2]

        if gr.test_scope(node):
            f.write("Solutie: ")
            node.show_path(f)
            # print("\n----------------\n")
            count -= 1
            if count == 0:
                show_stats(maxi, total, t1, f)
                return
        successors = gr.generate_successors(node, "euristica banala")
        for s in successors:
            index += 1
            c.put((s.g, index, s))
            total += 1
    if round(1000 * (time.time() - t1)) > timeout_limit:
        print("Timeout!")
        return


def a_star(gr, count, f, h_type="euristica banala"):
    f.write("\n\nA* " + "\n" + h_type + "\n\n")
    t1 = time.time()
    # in coada vom avea doar noduri de tip NodParcurgere (nodurile din arborele de parcurgere)
    nod = NodParcurgere(gr.start, None, 0, gr.get_h(gr.start, h_type))
    c = PriorityQueue()  # c va fi coada de prioritati
    index = 0  # index retine ordinea aparitiei nodurilor
    maxi = 0
    total = 1
    # este folosit pentru a ordona nodurile care au atat f, cat si g egale.
    c.put((nod.f, -nod.g, index, nod))  # insert in coada
    """
    Compararea se realizeaza astfel:
    1. Sunt comparate f-urile.
    2. Daca f-urile sunt egale, se compara g-urile.
    Cum ordonarea dupa g este descrescatoare, in tuplu va fi
    -nod.g.
    3. Daca si acestea sunt egale, se compara index-ul,
    pentru a exista o ordine.

    """
    while c:
        if round(1000 * (time.time() - t1)) > timeout_limit:
            print("Timeout!")
            return
        if not c.qsize():
            print("Nicio solutie!")
            return
        node = c.get()[3]  # get node
        maxi = max(maxi, c.qsize())
        if gr.test_scope(node):
            f.write("Solutie: ")
            node.show_path(f, show_cost=True, show_len=True)
            # print("\n----------------\n")
            # input()
            count -= 1
            if count == 0:
                show_stats(maxi, total, t1, f)
                return

        successors = gr.generate_successors(node, h_type)

        for s in successors:
            index += 1  # incrementam indexul
            c.put((s.f, -s.g, index, s))   # insert nodurile din lista
            total += 1


def ida_star(gr, nrSolutiiCautate, f, h_type="euristica banala"):
    global t1_ida
    f.write("\n\nIDA*" + "\n" + h_type + "\n\n")
    t1_ida = time.time()
    nodStart = NodParcurgere(gr.start, None, 0, gr.get_h(gr.start, h_type))
    limita = nodStart.f
    total = 0
    maxi = 0
    while True:
        # print("Limita de pornire: ", limita)
        nrSolutiiCautate, rez, aux = construieste_drum(gr, nodStart, limita, nrSolutiiCautate, f, total, h_type)
        maxi = max(maxi, aux)
        total += aux
        if rez == "gata":
            show_stats(maxi, total, t1_ida, f)
            if round(1000 * (time.time() - t1_ida)) > timeout_limit:
                print("Timeout!")
            break
        if rez == float('inf'):
            f.write("Nu exista solutii!")
            if round(1000 * (time.time() - t1_ida)) > timeout_limit:
                print("Timeout!")
            break
        limita = rez
        # print(">>> Limita noua: ", limita)
        # input()


def construieste_drum(gr, nodCurent, limita, nrSolutiiCautate, f, total=0, h_type="euristica banala"):
    global t1_ida
    # print("A ajuns la: ", nodCurent)
    if round(1000 * (time.time() - t1_ida)) > timeout_limit:
        print("Timeout!")
        return 0, "gata", total
    total += 1
    if nodCurent.f > limita:
        return nrSolutiiCautate, nodCurent.f, total
    if gr.test_scope(nodCurent) and nodCurent.f == limita:
        f.write("Solutie: ")
        nodCurent.show_path(f)
        # print(limita)
        # print("\n----------------\n")
        # input()
        nrSolutiiCautate -= 1
        if nrSolutiiCautate == 0:
            return 0, "gata", total
    lSuccesori = gr.generate_successors(nodCurent, h_type)
    minim = float('inf')
    for s in lSuccesori:
        nrSolutiiCautate, rez, total = construieste_drum(gr, s, limita, nrSolutiiCautate, f,  total, h_type)
        if rez == "gata":
            return 0, "gata", total
        # print("Compara ", rez, " cu ", minim)
        if rez < minim:
            minim = rez
            # print("Noul minim: ", minim)
    return nrSolutiiCautate, minim, total


def a_star_opt(gr, f, nrSolutiiCautate=1, h_type="euristica banala"):
    f.write("\nA*opt" + "\n" + h_type + "\n\n")
    t1 = time.time()
    # in coada vom avea doar noduri de tip NodParcurgere (nodurile din arborele de parcurgere)
    c = [NodParcurgere(gr.start, None, 0, gr.get_h(gr.start, h_type))]
    maxi = 0
    total = 1
    # lista open este c-ul
    # lista closed contine nodurile deja expandate
    closed = []
    while len(c) > 0:
        if round(1000 * (time.time() - t1)) > timeout_limit:
            print("Timeout!")
            return
        maxi = max(maxi, len(c))
        # print("Coada actuala: " + str(c))
        # input()
        nodCurent = c.pop(0)
        # adaug nodul in closed
        closed.append(nodCurent)
        if gr.test_scope(nodCurent):
            f.write("Solutie: ")
            nodCurent.show_path(f)
            # print("\n----------------\n")
            # input()

            nrSolutiiCautate -= 1
            if nrSolutiiCautate == 0:
                show_stats(maxi, total, t1, f)
                return
        lSuccesori = gr.generate_successors(nodCurent, h_type)
        # verific intai daca vreunul dintre succesori e deja in c(open)
        for s in lSuccesori:
            # print(s, "\n")
            gasitInC = False
            for i, elemC in enumerate(c):
                if np.array_equal(elemC.info, s.info):
                    gasitInC = True
                    if elemC.f <= s.f:
                        if s in lSuccesori:
                            lSuccesori.remove(s)
                        # break
                    else:
                        c.pop(i)
            if not gasitInC:
                for i, elemClosed in enumerate(closed):
                    if np.array_equal(elemClosed.info, s.info):
                        if elemClosed.f <= s.f:
                            if s in lSuccesori:
                                lSuccesori.remove(s)
                            # break
                        else:
                            closed.pop(i)

        for s in lSuccesori:
            i = 0
            total += 1
            found = False
            for i in range(len(c)):
                if c[i].f >= s.f:
                    found = True
                    break
            if found:
                c.insert(i, s)
            else:
                c.append(s)
    if round(1000 * (time.time() - t1)) > timeout_limit:
        print("Timeout!")
        return


folder_input = sys.argv[1]
folder_output = sys.argv[2]
nsol = int(sys.argv[3])
timeout_limit = int(sys.argv[4])

a_star_file = folder_input
# print(folder_input, folder_output, nsol, timeout_limit)

all_files = os.listdir(folder_input)
txt_files = list(filter(lambda x: x[-4:] == '.txt', all_files))

for file in txt_files:
    gr = Graph(folder_input + "\\" + file)
    output_file = "Output\output-" + file
    g = open(output_file, "w")
    a_star(gr, nsol, g, "euristica banala")
    a_star(gr, nsol, g, "e1")
    a_star(gr, nsol, g, "e2")
    a_star(gr, nsol, g, "neadmisibila")
    ucs(gr, g, nsol)
    ida_star(gr, nsol, g)
    ida_star(gr, nsol, g, "e1")
    ida_star(gr, nsol, g, "e2")
    a_star_opt(gr, g, nsol)
    a_star_opt(gr, g, nsol, "e1")
    a_star_opt(gr, g, nsol, "e2")
    print(file + " completed")
    g.close()




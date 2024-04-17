LANGUAGES = {"de", "en"}

RESPONSE_PATTERNS = {"de":
                         {"p1":'^anweisung: [^\n]+$',
                          "p2": '^\n*([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n*$',
                          "p1_terminate": '^\s*anweisung\s*:\s*fertig\s*$'},
                    "en":
                         {"p1":'^instruction: [^\n]+$',
                          "p2": '^\n*([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n([A-Z▢]\s){4}[A-Z▢]\n*$',
                          "p1_terminate": '^\s*instruction\s*:\s*done\s*$'},
                    "es":
                         {"p1":'',
                          "p2": '',
                          "p1_terminate": ''},
                    "ru":
                         {"p1":'',
                          "p2": '',
                          "p1_terminate": ''},
                    "tk":
                         {"p1":'',
                          "p2": '',
                          "p1_terminate": ''},
                    "tr":
                         {"p1":'',
                          "p2": '',
                          "p1_terminate": ''}
,
                    "te":
                         {"p1":'',
                          "p2": '',
                          "p1_terminate": ''}
                     }

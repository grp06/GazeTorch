class EYE:
    class LEFT:
        INNER_CORNER = 133
        OUTER_CORNER = 33
        TOP_CONTOUR = [133, 173, 157, 158, 159, 160, 161, 246, 33]
        BOTTOM_CONTOUR = [133, 155, 154, 153, 145, 144, 163, 7, 33]
        CONTOUR = TOP_CONTOUR[:-1] + BOTTOM_CONTOUR[-1:0:-1]

    class RIGHT:
        INNER_CORNER = 362
        OUTER_CORNER = 263
        TOP_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263]
        BOTTOM_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263]
        CONTOUR = TOP_CONTOUR[:-1] + BOTTOM_CONTOUR[-1:0:-1]

class OUTER:
    class LEFT:
        CONTOUR = [
            244, 189, 221, 222, 223, 224, 225, 113, 226,
            31, 228, 229, 230, 231, 232, 233
        ]
        TOP_CONTOUR = [189, 221, 222, 223, 224, 225, 113]
        BOTTOM_CONTOUR = [31, 228, 229, 230, 231, 232, 233]

    class RIGHT:
        CONTOUR = [
            464, 413, 441, 442, 443, 444, 445, 342, 446,
            261, 448, 449, 450, 451, 452, 453
        ]
        TOP_CONTOUR = [413, 441, 442, 443, 444, 445, 342]
        BOTTOM_CONTOUR = [261, 448, 449, 450, 451, 452, 453]

    ALL = LEFT.CONTOUR + RIGHT.CONTOUR # 32x2

class EYELID:
    LEFT_CONTOUR = OUTER.LEFT.TOP_CONTOUR[:9] + [130] + EYE.LEFT.TOP_CONTOUR[::-1] + [243]
    RIGHT_CONTOUR = OUTER.RIGHT.TOP_CONTOUR[:9] + [359] + EYE.RIGHT.TOP_CONTOUR[::-1] + [463]

class EYEBROW:
    LEFT = [46, 53, 52, 65, 55] # OUTER-TO-INNER
    LEFT_TOP = [70, 63, 105, 66, 107]
    RIGHT = [276, 283, 282, 295, 285]
    RIGHT_TOP = [300, 293, 334, 296, 336]

class NOSE:
    TIP = 4
    TIP_LINE = [48, 115, 220, 45, 4, 275, 440, 344, 278]

FACEMASK = [123, 352, 198, 420, 195, 136, 365, 152, 36, 266, 170, 395]
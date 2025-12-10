"""
utils.excels の Docstring
"""
class DiffHighlighter:
    def __init__(self):
        pass

    # ---------- 空白除去して比較する関数（s1 を GT として扱う） ----------
    def make_diff_masks(self, s1, s2, s3):
        """
        s1: GT （常に黒）
        s2: 比較対象1（GTと違う文字だけ True）
        s3: 比較対象2（GTと違う文字だけ True）
        """

        def strip_spaces_with_map(s):
            no_spaces = []
            mapping = []
            for i, ch in enumerate(s):
                if ch in (" ", "　"):
                    continue
                no_spaces.append(ch)
                mapping.append(i)
            return "".join(no_spaces), mapping

        # 空白除去版とマッピング
        gt_ns,  gt_map  = strip_spaces_with_map(s1)
        s2_ns,  s2_map  = strip_spaces_with_map(s2)
        s3_ns,  s3_map  = strip_spaces_with_map(s3)

        # GT 基準で比較するので、GT 側のマスクは常に False
        gt_ns_mask  = [False] * len(gt_ns)
        s2_ns_mask  = [False] * len(s2_ns)
        s3_ns_mask  = [False] * len(s3_ns)

        # 1) GT の長さまで見比べる
        for i, gch in enumerate(gt_ns):
            # s2 との比較
            if i < len(s2_ns):
                if s2_ns[i] != gch:
                    s2_ns_mask[i] = True
            # s3 との比較
            if i < len(s3_ns):
                if s3_ns[i] != gch:
                    s3_ns_mask[i] = True

        # 2) GT より長い部分（あふれている文字）は GT には存在しないので全部 True
        for i in range(len(gt_ns), len(s2_ns)):
            s2_ns_mask[i] = True
        for i in range(len(gt_ns), len(s3_ns)):
            s3_ns_mask[i] = True

        # ---- 空白ありのマスクに変換 ----
        mask1 = [False] * len(s1)  # GT は全部 False のまま
        mask2 = [False] * len(s2)
        mask3 = [False] * len(s3)

        # s2: no-space マスクを元の位置に戻す
        for ns_i, orig_i in enumerate(s2_map):
            if s2_ns_mask[ns_i]:
                mask2[orig_i] = True

        # s3: 同様に戻す
        for ns_i, orig_i in enumerate(s3_map):
            if s3_ns_mask[ns_i]:
                mask3[orig_i] = True

        return mask1, mask2, mask3

    # ---------- Excel へ反映する（1文字ずつセルに書く・openpyxl用） ----------
    def write_string(self, ws, row, col_start, text, mask, font_red):
        for i, (ch, flag) in enumerate(zip(text, mask)):
            cell = ws.cell(row=row, column=col_start + i)
            cell.value = ch
            if flag and ch not in (" ", "　"):
                cell.font = font_red

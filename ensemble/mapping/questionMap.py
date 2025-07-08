def extract_entities(text: str):
    text = text.lower()

    # Deteksi waktu
    if "hari ini" in text or "sekarang" in text:
        waktu = "hari_ini"
    elif "kemarin" in text:
        waktu = "kemarin"
    elif "minggu ini" in text:
        waktu = "minggu_ini"
    elif "bulan ini" in text:
        waktu = "bulan_ini"
    elif "tahun ini" in text:
        waktu = "tahun_ini"
    else:
        waktu = "all"

    # Deteksi target
    if "modal" in text:
        target = "modal"
    elif "rugi" in text or "kerugian" in text:
        target = "rugi"
    elif "untung" in text or "profit" in text or "laba" in text or "keuntungan" in text:
        target = "profit"
    elif "perbandingan" in text or "toko a" in text or "toko b" in text:
        target = "compare"
    else:
        target = "unknown"

    return waktu, target

# Contoh manual run
if __name__ == "__main__":
    example = "Berapa keuntungan saya bulan ini?"
    waktu, target = extract_entities(example)
    print(f"Intent terdeteksi:\n- Waktu: {waktu}\n- Target: {target}")

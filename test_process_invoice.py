from pathlib import Path

from windows_native import config_store, mcp_excel_server


def main() -> None:
    # Pfad zu deiner Testrechnung im Incoming-Ordner
    invoice_path = config_store.get_incoming_dir() / "invoice.pdf"

    result = mcp_excel_server.process_invoice_file(str(invoice_path)) # the path needs to be stringified because a pathlib object is not a string
    print("Result from process_invoice_file:")
    print(result)


if __name__ == "__main__": # __name__ is a special variable; if directly run, it will be __main__; otherwise if imported, it would be test_process_invoice
    main()



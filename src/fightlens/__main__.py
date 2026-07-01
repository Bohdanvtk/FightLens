from fightlens.gemini import generate_text


def main() -> None:
    response = generate_text(
        ""
    )

    print(response)


if __name__ == "__main__":
    main()
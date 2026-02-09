import logging
import sys
from typing import Optional


def configurar_logging(
    nivel: int = logging.INFO,
    formato: Optional[str] = None
) -> None:
    """
    Configura o logging global da aplicação.

    :param nivel: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param formato: Formato opcional das mensagens de log
    """
    formato_padrao = (
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logging.basicConfig(
        level=nivel,
        format=formato or formato_padrao,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


import sys
sys.path.append('../../pyquickhelper/src')
sys.path.append('../../jyquickhelper/src')
sys.path.append('../../lightmlrestapi/src')
sys.path.append('../../mlinstights/src')
sys.path.append('../../pyensae/src')
sys.path.append('../../mlinsights/src')
sys.path.append('../../pandas_streaming/src')
sys.path.append('../../ensae_projects/src')


def process_server(host, port):
    import logging
    logger = logging.getLogger('search_images_dogcat')
    logger.setLevel(logging.INFO)
    hdlr = logging.FileHandler('search_images_dogcat.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)     

    from ensae_projects.restapi import search_images_dogcat
    app = search_images_dogcat()

    from waitress import serve
    serve(app, host=host, port=port)


process_server('127.0.0.1', 8081)

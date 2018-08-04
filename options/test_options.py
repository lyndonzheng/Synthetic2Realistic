from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help = 'saves results here')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

        self.isTrain = False
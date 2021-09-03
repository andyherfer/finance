def __print_arg(arg):
    if arg is not None:
        return arg
    else:
        return 'MISSING'
    
    
class BondCalculator:
    def __init__(
        self, 
        yield_to_maturity=None,
        current_price=None,
        nominal_value=1000, 
        years=None,
        period=1,
        n_coupons=None,
        coupon_rate=None,
        coupon_value=None,
                ):
        self.yield_to_maturity = yield_to_maturity
        self.current_price = current_price
        self.nominal_value = nominal_value 
        self.years = years
        self.period = period
        self.n_coupons = n_coupons
        self.coupon_rate = coupon_rate
        self.coupon_value = coupon_value
        
    def __str__(self):
        """Prints the state of the calculator"""
        string = f"""BondCalculator(
        yield_to_maturity={__print_arg(self.yield_to_maturity)},
        current_price={__print_arg(self.current_price)},
        nominal_value={__print_arg(self.nominal_value)}, 
        years={__print_arg(self.years)},
        period={__print_arg(self.period)},
        n_coupons={__print_arg(self.n_coupons)},
        coupon_rate={__print_arg(self.coupon_rate)},
        coupon_value={__print_arg(self.coupon_value)},
                )"""
        return string
        
    def get_period(self):
        if self.period is None:
            self.period = self.years / self.get_n_coupons()
        return self.period
    
    def get_n_coupons(self):
        if self.n_coupons is None:
            self.n_coupons = self.get_period() * self.years
        return self.n_coupons
            
    def get_coupon_rate(self):
        if self.coupon_rate is None:
            self.coupon_rate = (self.get_coupon_value() / self.get_nominal_value()) * self.get_period()
                
        return self.coupon_rate
    
    def get_coupon_value(self):
        if self.coupon_value is None:
            if self.coupon_rate is not None:
                self.coupon_value = self.get_nominal_value() * self.get_coupon_rate()/self.get_period()
            else:
                self.coupon_value = (self.get_current_price()-self._get_right())/self._get_parenthesis()
        return self.coupon_value
    
    def get_nominal_value(self):
        if self.nominal_value is None:
            self.nominal_value = self.get_coupon_value() / self.get_coupon_rate()/self.get_period()
        return self.nominal_value
    
    def _get_right(self):
        return self.get_nominal_value()/(1 + self.yield_to_maturity/self.period)**self.get_n_coupons()
    
    def _get_parenthesis(self):
        top = 1 - 1/(1 + self.yield_to_maturity/self.get_period())**self.get_n_coupons()
        bottom = self.yield_to_maturity / self.get_period()
        return top/bottom
    
    def get_current_price(self):
        if self.current_price is None:
            self.current_price = self.get_coupon_value()*self._get_parenthesis() + self._get_right()
        return self.current_price
        
   
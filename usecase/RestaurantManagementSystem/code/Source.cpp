#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

//This is the latest we got
using namespace std;
#define SUCCESS 1;
#define FAIL 0;

class Item {
private:
	/* Each item object has price , tax , total price , type (0 for drink and 1 for food),
	stockState ( 0 for out of stock and 1 for available , itemID & itemname */
	float price, tax = ((price * 14) / 100), totalPrice = price + tax;
	bool type, stockState;
	int itemID;
	string itemname;

public:

	/* A function to set the attributes of items , it take input itemtype , price , stockstate , itemit, itemname */
	void setItem(bool itemType, float itemPrice, bool itemStockState, int ItemID, string itemName)
	{
		type = itemType;
		price = itemPrice;
		stockState = itemStockState;
		itemID = ItemID;
		itemname = itemName;
	}
	/* The following are functions to view the item details */
	float get_price()
	{
		return price;
	}
	bool get_type()
	{
		return type;
	}
	bool get_stockState()
	{
		return stockState;
	}
	float get_totalPrice()
	{
		return totalPrice;
	}
	int get_itemID()
	{
		return itemID;
	}
	string get_itemname()
	{
		return itemname;
	}
	/* Operator Overloaded to compare between two item objects */
	bool operator == (Item item1)
	{
		if ((item1.get_itemID() == itemID) && ((item1.get_itemname()) == itemname) \
			&& (item1.get_price() == price) && (item1.get_stockState() == stockState)\
			&& (item1.get_type() == type))
		{

			return 1;
		}
		else
		{
			return 0;
		}
	}

};

class Menu {
public:
	/* The Menu class will be a vector of pointers to Item class objects */
	vector<Item*> MainMenu;

	/* the addItem method can be used to add an item to the vectors by passing the Item object address */
	void addItem(Item* myItem)
	{
		MainMenu.push_back(myItem);
	}
	/* Deleting an item will be done by sending the required item ID to the function deleteItem */
	void deleteItem(int itemId)
	{
		/* A loop iterating over the vector elements */
		for (unsigned int i = 0; i < MainMenu.size(); i++)
		{
			/* the item will be deleted if its ID is the same as the (required item to be deleted) ID,
				, the arrow operator is used to access the attribute(ID) of the current iterated object */
			if ((MainMenu[i]->get_itemID()) == itemId)
			{
				MainMenu.erase(MainMenu.begin() + i);
			}
		}
	}

};

/*Payment class and its inheritance, Order*/

class Payment
{

private:
	float price;
	bool paymentStatus;

public:
	void setPaymentStatus(bool ps)
	{
		paymentStatus = ps;
	}

	bool  getPaymentStatus()
	{
		return paymentStatus;
	}


	float getPrice()
	{
		return price;
	}

	virtual bool checkCharge(string cardID, float price, float charge) = 0;
};

class Credit : public Payment
{
public:
	bool checkCardValidity(string cardID, string pw)
	{
		/*The card info will contain id no., password, balance(Charge)*/
		//cout << "hiii, I'm in" << endl;
		ifstream readBalance("visa.txt");
		string card_id;
		string card_pw;
		float card_balance;

		while (readBalance >> card_id >> card_pw >> card_balance)
		{
			//cout << "I'm in 1" << endl;
			if ((cardID == card_id) && (pw == card_pw))
			{
				//cout << "I'm in" << endl;
				return 1;
			}

			if (!((cardID == card_id) && (pw == card_pw)))
			{
				return 0;
			}
		}


	}

	bool checkCharge(string cardID, float price, float charge)
	{

		/* An object must be instantiated from credit class to get its price using the getPrice function to use the price in this current function
		like so :
		Credit my_payment;
		price = my_payment.getPrice();
		*/
		ifstream readBalance("visa.txt");
		string card_id;
		string card_pw;
		float card_balance;

		while (readBalance >> card_id >> card_pw >> card_balance)
		{
			if (cardID == card_id)
			{
				if (card_balance >= price)
				{
					return true;
				}

				else if (card_balance < price)
				{
					return false;
				}
			}


		}

	}
};

class Cash : public Payment
{

	bool checkCharge(float price, float charge)
	{
		if (charge >= price)
		{
			return true;
		}

		else if (charge < price)
		{
			return false;
		}
	}

};

/*Order class*/
class Order
{
private:
	bool orderType;
	float orderPrice = 0.0;
	float totalPrice = orderPrice + 0.14 * orderPrice;
	bool orderStatus = 0;
	vector <Item> itemSelectedVect;
	int numberOfItem;
public:

	//this function add item and number of items chosen from this specific item
	void addToOrder(Item newItem, int selectedItemNumber = 1)
	{
		for (int i = 0; i < selectedItemNumber; i++)
		{
			itemSelectedVect.push_back(newItem);
		}
	}

	void removeFromOrder(Item toBeRemovedItem, int selectedItemNumber)
	{
		/*vector<Item>::iterator it;
		it = find(itemSelectedVect.begin(), itemSelectedVect.end(), toBeRemovedItem);
		if (it != itemSelectedVect.end())
		{
			index = it - itemSelectedVect.begin();
		}
		for (int i = 0; i < itemSelectedVect.size(); i++)
		{
			itemSelectedVect.erase(itemSelectedVect.begin()+index);
		}
		/*ptrdiff_t pos = find(itemSelectedVect.begin(), itemSelectedVect.end(), toBeRemovedItem) - itemSelectedVect.begin();
		for (int i = 0; i < itemSelectedVect.size(); i++)
		{
			itemSelectedVect.erase(itemSelectedVect.begin() + pos);
		}
		*/

		// PLEASE Return to the main function for a more detailed explanation if I was asleep , and delete this comment once you  read it
		// msh 3yzyn nesalem el  proj keda XD
		for (int i = 0; i < (itemSelectedVect.size()); i++)
		{

			if (itemSelectedVect[i] == toBeRemovedItem)
			{
				itemSelectedVect.erase(itemSelectedVect.begin() + i);
			}

		}
	}

	float confirmOrder()
	{
		for (int i = 0; i < itemSelectedVect.size(); i++)
		{
			orderPrice += (itemSelectedVect[i]).get_price();
		}
		return totalPrice;
	}

	/*void setOrderStatus(bool stat)
	{
		orderStatus = stat;
	}*/

	bool getOrderStatus()
	{
		return orderStatus;
	}

	friend class Admin;
	friend class System;
};

/*Table Class*/
class Table
{
private:
	bool stat;
public:
	vector<bool> tables_status;
	void tableReservation()
	{
		//here I'm pushing (initializing) table_status for six tables
		//1 for available ------ 0 for reserved
		//vector<bool> tables_status;
		//t1_status
		tables_status.push_back(1);

		//t2_status
		tables_status.push_back(0);

		//t3_status
		tables_status.push_back(1);

		//t4_status
		tables_status.push_back(1);

		//t5_status
		tables_status.push_back(0);

		//t6_status
		tables_status.push_back(0);

		//t7_status
		tables_status.push_back(1);

		//t8_status
		tables_status.push_back(1);
	}

	//this is a constructor that will initialize the table status
	Table()
	{

		tableReservation();

		/*for (int i = 0; i < tables_status.size(); i++)
		{
			cout << "Table " << i + 1 << " status is:" << tables_status[i] << endl;
		}*/

	}

	void reserveTable(int table_no)
	{
		bool table_stat = tables_status.at(table_no - 1);

		//cout << table_no << " " << tables_status[table_no - 1] << endl;

		//here we need to check if the table is already reserved

		//if (table_stat == 0)
		//{
		//	//cout << "Am I in?" << endl;

		//	return 0; //reservation failed choose another table
		//}

		if (table_stat == 1)
		{
			//here we will change the value of the table status in the vector
			tables_status[table_no - 1] = 0;

			//cout << "changing the table stat from " << !tables_status[table_no - 1] << " to " << tables_status[table_no - 1] << endl;
			//return 1; //table reserved
		}
	}

	friend class System;
};


class User
{
protected:
	string name, address, city, number;
	int age, id;
	bool is_logged_in;
public:
	/*User Class Code Goes Here*/
	void setName(string myName)
	{
		name = myName;
	}
	void setAge(int myAge)
	{
		age = myAge;
	}
	void setAddress(string myAdddress)
	{
		address = myAdddress;
	}
	void setCity(string myCity)
	{
		city = myCity;
	}
	void setNumber(string myNumber)
	{
		number = myNumber;
	}
	void setID(int myID)
	{
		id = myID;
	}
	/* The get functions should start here */
	string getName()
	{
		return name;
	}
	virtual void set_login() = 0;
	virtual void reset_login() = 0;
	int getAge()
	{
		return age;
	}

	string getAddress()
	{
		return address;
	}

	string getCity()
	{
		return city;
	}

	string getNumber()
	{
		return number;
	}

	int getID()
	{
		return id;
	}
};

/*system class*/
class System
{
public:
	vector <Order*> AllOrdersVect;
	Order customizedOrder;
	Menu integratedMenu;
	Table tablesArr;
	void RegisterOrder(Order* myOrder)
	{
		AllOrdersVect.push_back(myOrder);
	}
	void makeOrder(Item item)
	{
		customizedOrder.addToOrder(item);
	}
	void viewMenu(Menu myMenu)
	{
		for (int i = 0; i < myMenu.MainMenu.size(); i++)
		{
			cout << myMenu.MainMenu[i] << endl;
		}
	}
	void makePayment(Credit mypay, float myPrice, string card_ID, string pw_, float charge)
	{
		//this needs to be reviewed *URGENTLY*
		bool validity = mypay.checkCardValidity(card_ID, pw_);
		bool valid_charge = mypay.checkCharge(card_ID, myPrice, charge);
	}
	string checkPersonalOrderStatus(Order& my_order)
	{
		if (!(my_order.getOrderStatus()))
		{
			return "Your order is not ready :'( !";
		}
		if ((my_order.getOrderStatus()))
		{
			return "Your order is ready, Bon apetite!";
		}
	}
	void checkAvailableTables()
	{
		vector <int>availableTables;
		int index = 0;

		for (int i = 0; i < tablesArr.tables_status.size(); i++)
		{
			if (tablesArr.tables_status[i] == 1)
			{
				availableTables.push_back(i);
			}
		}

		/*testing*/
	   /*for (int j = 0; j < availableTables.size(); j++)
	   {
		   cout << "This table is " << j << " and its no. is " << availableTables[j] + 1 << endl;
	   }*/



	}
};


class Admin : public User
{
private:
	string jobType;
	bool workStatus, active;
	float rate;

public:
	Admin()
	{
		setAddress("maadi"); setAge(10); setCity("Cairo"); setID(007); setName("None"); setNumber("007"); setWorkStatus(1);
	}
	void setWorkStatus(bool mystatus)
	{
		workStatus = mystatus;
	}
	bool getWorkStatus()
	{
		return workStatus;
	}
	void editMenu(Item* myItem, Menu* myMenu)
	{
		myMenu->addItem(myItem);
	}
	bool setStatus()
	{
		active = 1;
	}
	void setOrderStatus(Order& order)
	{
		order.orderStatus = 1;
	}

	void checkAllOrderStatus(System MainSystemObject)
	{
		for (int i = 0; i < MainSystemObject.AllOrdersVect.size(); i++)
		{
			cout << "Order (" << i + 1 << ") contains : " << endl;
			Order current_order = *(MainSystemObject.AllOrdersVect[i]);
			for (int j = 0; j < current_order.itemSelectedVect.size(); j++)
			{
				cout << "\t" << current_order.itemSelectedVect[j].get_itemname() << " , Status : " << current_order.orderStatus << endl;
			}
		}
	}
	void set_login()
	{
		is_logged_in = 1;
	}
	void reset_login()
	{
		is_logged_in = 0;
	}
	bool view_loginStatus()
	{
		return is_logged_in;
	}
};

class Account
{
private:
	string password;
	Admin AdminAccount;
public:
	string view_password()
	{
		return password;
	}
	void change_password(string newpassword)
	{
		password = newpassword;
	}
	Admin* get_Admin()
	{
		return &AdminAccount;
	}
	bool newAdminAccount(Admin myAdminAcc, string myPassword)
	{
		AdminAccount = myAdminAcc;
		password = myPassword;
		return SUCCESS;
	}

	bool deleteAccount(int myID, string myPassword, Account* myAccAdmin)
	{
		
		if (myPassword == password)
		{
			delete myAccAdmin;
			return SUCCESS;
		}
		else
		{
			return FAIL;
		}
	}

	bool changePassword(int myID, string myOldPassword, string myNewPassword, Account* myAdmin)
	{
		if (myOldPassword == myAdmin->view_password())
		{
			cout << "Current Password" << myAdmin->view_password() << endl;
			myAdmin->change_password(myNewPassword);
			cout << "Current Password" << myAdmin->view_password()<<endl;
			return SUCCESS;
		}
		else
		{
			//acout << myOldPassword << endl << myAdmin->password << endl;
			return FAIL;
		}
	}
	void viewAccountDetails(int myID, string  myPassword, Account* myAdminAcc)
	{
		
		if ((myAdminAcc->get_Admin())->getID() == myID && myAdminAcc->view_password() == myPassword )
		{
			cout << myAdminAcc->get_Admin()->getAddress() << endl;
			cout << myAdminAcc->get_Admin()->getAge() << endl;
			cout << myAdminAcc->get_Admin()->getCity() << endl;
			cout << myAdminAcc->get_Admin()->getID() << endl;
			cout << myAdminAcc->get_Admin()->getName() << endl;
			cout << myAdminAcc->get_Admin()->getNumber() << endl;
			cout << myAdminAcc->get_Admin()->getWorkStatus() << endl;
		}
		else
		{
			cout << "\033[1;31mWrong Cridentials Provided\033[0m\n";
		}

	}

	bool login(int myID, string myPassword, Admin* myadmin)
	{
		myadmin->set_login();
		return SUCCESS;
	}

	bool logout(Admin* myadmin)
	{
		myadmin->reset_login();
		return SUCCESS;
	}
};
void bankSimulation()
{
	/*initializing some credit card info*/
	ofstream balance("visa.txt", ios::app);
	string card1_No = "4024007141525864";
	string card1_pw = "dina";
	float card1_balance = 100;

	string card2_No = "4532295627154748";
	string card2_pw = "dinadina";
	float card2_balance = 1;

	string card3_No = "4929230781795532";
	string card3_pw = "dinadinadina";
	float card3_balance = 200;

	balance << card1_No << " " << card1_pw << " " << card1_balance << endl;
	//cout << card1_No << " " << card1_pw << endl;

	balance << card2_No << " " << card2_pw << " " << card2_balance << endl;
	//cout << card2_No << " " << card2_pw << endl;

	balance << card3_No << " " << card3_pw << " " << card3_balance << endl;
	//cout << card3_No << " " << card3_pw << endl;
	balance.close();
}




int main() {
	/*											START OF MY TEST CASE												*/
	/* Creating a main system object to act as a link between our different objects */
	System MainSystem;
	Admin Dina;
	Dina.setName("Dina");
	Account RestaurantManager;
	/* Creating a new Admin Account with password "Character" */
	RestaurantManager.newAdminAccount(Dina, "character");
	/* Requesting to view the account details */
	RestaurantManager.viewAccountDetails(Dina.getID(), "character", &RestaurantManager);
	/* Changing password of that Account */
	RestaurantManager.changePassword(Dina.getID(), "character", "new", &RestaurantManager);
	/* Trying to view account details using same old password , the request will be denied */
	RestaurantManager.viewAccountDetails(Dina.getID(), "character", &RestaurantManager);
	/* Now trying to access with the new password */
	RestaurantManager.viewAccountDetails(Dina.getID(), "new", &RestaurantManager);
	/* Creating a couple of items to add to Menu */
	Item Pepsi;
	Pepsi.setItem(0, 5, 1, 777, "Pepsi Cola");
	Item CheeseCake;
	CheeseCake.setItem(1, 50, 1, 778, "CheeseCake Desert");
	Item Tea;
	Tea.setItem(0, 10, 1, 779, "Shay");
	Item Pasta;
	Pasta.setItem(1, 40, 1, 780, "Pasta Avec de sauce francaise");
	/* Adding items to the menu */
	Menu Main;
	Main.addItem(&Pepsi); Main.addItem(&CheeseCake); Main.addItem(&Pasta); Main.addItem(&Tea);
	/* Checking The Menu Content */
	for (int i = 0; i < Main.MainMenu.size(); i++) cout << (Main.MainMenu[i])->get_itemname() << "  :  ";
	cout << endl;
	/* Removing item using ID from the Menu */
	Main.deleteItem(779);
	for (int i = 0; i < Main.MainMenu.size(); i++) cout << (Main.MainMenu[i])->get_itemname() << "  :  ";
	cout << endl;
	/* Taking Order From Customer */
	Order First_order;
	First_order.addToOrder(Pepsi, 1);
	First_order.addToOrder(CheeseCake, 2);
	/*Registering Order in the main system */
	MainSystem.RegisterOrder(&First_order);
	Order Second_order;
	Second_order.addToOrder(Pasta, 3);
	Second_order.addToOrder(CheeseCake, 2);
	/*Registering Order in the main system */
	MainSystem.RegisterOrder(&Second_order);
	/* Now Checking the order status of each order registered on the system , this can only be done by the adming account */
	/*						Status 0 indicates not ready , Status 1 indicates ready										  */
	Dina.checkAllOrderStatus(MainSystem);

	/*									HERE ENDS MY TEST CASE																*/		


	///*Test Case for checking order status*/
	//System james_bond;
	//Order my_order;
	//bool orderStat = my_order.getOrderStatus();
	//cout << "Your order is (boolen): " << orderStat << endl;
	//string result = james_bond.checkPersonalOrderStatus(my_order);
	//cout << result << endl << endl;
	//Dina.setOrderStatus(my_order);
	//orderStat = my_order.getOrderStatus();
	//cout << "Your order now is (boolen): " << orderStat << endl;
	//result = james_bond.checkPersonalOrderStatus(my_order);
	//cout << result << endl;
	//cout << endl;

	/* Now to users part , note  that we can't create users as it is an abstract class which cannot be objectified */

		/*This test case for check available tables*/
		/*System dina;
		dina.checkAvailableTables();*/

		/*Item item1;
		item1.setItem(0, 0, 0, 12, "Amr");
		Item item2;
		item2.setItem(0, 0, 0, 10, "Dina");
		if (item1 == item1)
		{
			cout << "Succeeded";
		}
		else
		{
			cout << "Fail";
		}*/
		//bankSimulation(); //This only initialized one time to fill the credit card info, or you could either remove ios::app to overwrite the file


		/*Test case
		Table dina;
		bool my_result = dina.reserveTable(2);
		cout << "your result is :" << my_result << endl;*/

		/*test case
		Credit my_payment;
		float price = my_payment.getPrice();
		string card1 = "4024007141525864";
		string pw1 = "dina";
		bool result;
		result = my_payment.checkCardValidity(card1, pw1);
		cout << "My result is " << result << endl;*/

	return 0;
}
